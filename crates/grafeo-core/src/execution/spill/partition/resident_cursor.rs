//! Consuming drain for a partition that never acquired a spill base.

use super::{
    AccountedError, AccountedPartitionKey, AccountedPartitionValue, MemoryGrantError,
    PartitionDrainEntry, PartitionEntry, PartitionFailureCleanup, PartitionOperationError,
    PartitionWorkspace, PartitionedState, SerializedKey, check_cancellation, new_partition_map,
    partition_memory_error,
};

/// The state's root continues to cover this moved map, including its buckets,
/// until each physical allocation is destroyed or transferred to an output.
pub(super) struct ResidentPartition<V> {
    index: usize,
    entries: Option<hashbrown::hash_map::IntoIter<SerializedKey, PartitionEntry<V>>>,
    // A fallible callback must leave its value here for terminal cleanup. A raw
    // local V could panic again while the callback's panic is unwinding.
    pending: Option<(SerializedKey, PartitionEntry<V>)>,
    map_bytes: usize,
    cleanup: Option<AccountedError>,
}

impl<V: Clone + Send + Sync + 'static> ResidentPartition<V> {
    pub(super) fn begin(state: &mut PartitionedState<V>, index: usize) -> Self {
        let partition = state.partitions[index]
            .take()
            .unwrap_or_else(new_partition_map);
        let map_bytes = PartitionedState::<V>::partition_map_allocation_bytes(&partition);
        Self {
            index,
            entries: Some(partition.into_iter()),
            pending: None,
            map_bytes,
            cleanup: state.failure_cleanup.clone(),
        }
    }

    pub(super) fn next(
        &mut self,
        state: &mut PartitionedState<V>,
    ) -> Result<Option<PartitionDrainEntry<V>>, PartitionOperationError> {
        check_cancellation(state.cancellation.as_ref())?;
        if state.grant.is_none() {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain has no root authority",
            });
        }
        if self.pending.is_none() {
            self.pending = self.entries.as_mut().and_then(Iterator::next);
        }
        let Some((serialized, entry)) = self.pending.as_ref() else {
            if state.partition_sizes[self.index] != 0 {
                return Err(PartitionOperationError::NativeMapInvariant {
                    message: "resident drain count disagrees with its catalog",
                });
            }
            let root = state
                .grant
                .as_mut()
                .ok_or(PartitionOperationError::NativeMapInvariant {
                    message: "resident drain has no root authority",
                })?;
            let map_grant =
                root.split(self.map_bytes)
                    .ok_or(PartitionOperationError::NativeMapInvariant {
                        message: "resident drain map lost its root authority",
                    })?;
            // Even an exhausted IntoIter still owns the native buckets.
            // Check and retain its child before destroying those buckets.
            drop(self.entries.take());
            self.map_bytes = 0;
            drop(map_grant);
            state.partitions[self.index] = Some(new_partition_map());
            return Ok(None);
        };

        let remaining = state.partition_sizes[self.index].checked_sub(1).ok_or(
            PartitionOperationError::NativeMapInvariant {
                message: "resident drain contains an uncatalogued entry",
            },
        )?;
        let observed =
            (state.value_resident_capacity)(&entry.value).map_err(partition_memory_error)?;
        if observed > entry.resident_bound {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain value exceeds its stored bound",
            });
        }
        check_cancellation(state.cancellation.as_ref())?;
        let serialized_bytes = serialized.0.capacity();
        let transferred = observed.checked_add(serialized_bytes).ok_or_else(|| {
            partition_memory_error(MemoryGrantError::ArithmeticOverflow {
                current_bytes: observed,
                additional_bytes: serialized_bytes,
            })
        })?;
        let root = state
            .grant
            .as_ref()
            .ok_or(PartitionOperationError::NativeMapInvariant {
                message: "resident drain has no root authority",
            })?;
        if root.size() < transferred {
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain entry lost its root authority",
            });
        }
        let decoded_bound = PartitionedState::<V>::decoded_key_resident_bound(serialized.0.len())
            .map_err(partition_memory_error)?;
        let mut value_workspace = PartitionWorkspace::new(
            state.split_workspace_grant(entry.resident_bound - observed)?,
            self.cleanup.as_ref(),
        );
        let key_workspace = PartitionWorkspace::new(
            state.split_workspace_grant(decoded_bound)?,
            self.cleanup.as_ref(),
        );
        let key = serialized.to_values(entry.num_key_columns, state.frame_limits)?;
        check_cancellation(state.cancellation.as_ref())?;
        let key = AccountedPartitionKey::new(key, key_workspace.into_grant()?);

        // Admission and opaque callbacks are finished. Move the existing value
        // authority rather than admitting a duplicate copy of its heap.
        let root = state
            .grant
            .as_mut()
            .ok_or(PartitionOperationError::NativeMapInvariant {
                message: "resident drain has no root authority",
            })?;
        let value_grant = value_workspace.grant_mut()?;
        let resident_grant =
            root.split(observed)
                .ok_or(PartitionOperationError::NativeMapInvariant {
                    message: "resident drain value lost its root authority",
                })?;
        if let Err(grant) = value_grant.try_merge(resident_grant) {
            // Preserve the root's coverage of pending on the invariant path.
            if let Err(grant) = root.try_merge(grant) {
                if let Some(cleanup) = &self.cleanup {
                    cleanup.inspect::<PartitionFailureCleanup, _>(
                        PartitionFailureCleanup::mark_failed,
                    );
                }
                std::mem::forget(grant);
            }
            return Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain child lost its root identity",
            });
        }
        let value_grant = value_workspace.into_grant()?;
        let (serialized, entry) =
            self.pending
                .take()
                .ok_or(PartitionOperationError::NativeMapInvariant {
                    message: "resident drain lost its admitted entry",
                })?;
        let output = PartitionDrainEntry {
            key,
            value: AccountedPartitionValue::new(entry.value, value_grant),
        };
        drop(serialized);
        drop(
            root.split(serialized_bytes)
                .ok_or(PartitionOperationError::NativeMapInvariant {
                    message: "resident drain key lost its root authority",
                })?,
        );
        state.partition_sizes[self.index] = remaining;
        Ok(Some(output))
    }
}

impl<V> ResidentPartition<V> {
    /// The caller retains the root throughout this operation and must keep it
    /// charged fail-closed when any physical destructor failed.
    pub(super) fn destroy(mut self) -> bool {
        self.destroy_inner()
    }

    fn destroy_inner(&mut self) -> bool {
        let mut destroyed = true;
        if let Some(entry) = self.pending.take() {
            destroyed &= super::super::run_cleanup_backstop(|| {
                drop(entry);
                Ok::<(), std::convert::Infallible>(())
            });
        }
        if let Some(mut entries) = self.entries.take() {
            // Drop each opaque V under its own backstop. Dropping the complete
            // iterator could double-panic when two remaining values are hostile.
            for entry in entries.by_ref() {
                destroyed &= super::super::run_cleanup_backstop(|| {
                    drop(entry);
                    Ok::<(), std::convert::Infallible>(())
                });
            }
            drop(entries);
        }
        if !destroyed && let Some(cleanup) = &self.cleanup {
            cleanup.inspect::<PartitionFailureCleanup, _>(PartitionFailureCleanup::mark_failed);
        }
        destroyed
    }
}

impl<V> Drop for ResidentPartition<V> {
    fn drop(&mut self) {
        self.destroy_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::QueryExecutionControl;
    use crate::execution::spill::{BorrowedSpillFixture, SpillManager};
    use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig, MemoryRegion};
    use grafeo_common::types::Value;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tempfile::TempDir;

    struct Probe {
        bytes: Vec<u8>,
        memory: Arc<BufferManager>,
        dropped: Arc<AtomicUsize>,
        allocated_at_drop: Arc<AtomicUsize>,
        hostile_drop: Arc<AtomicBool>,
    }

    impl Clone for Probe {
        fn clone(&self) -> Self {
            panic!("resident drain must move its value without cloning")
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            self.allocated_at_drop
                .store(self.memory.allocated(), Ordering::Release);
            self.dropped.fetch_add(1, Ordering::AcqRel);
            assert!(
                !self.hostile_drop.load(Ordering::Acquire),
                "hostile resident value destructor"
            );
        }
    }

    struct Fixture {
        state: PartitionedState<Probe>,
        memory: Arc<BufferManager>,
        manager: Arc<SpillManager>,
        dropped: Arc<AtomicUsize>,
        allocated_at_drop: Arc<AtomicUsize>,
        hostile_drop: Arc<AtomicBool>,
        hostile_capacity: Arc<AtomicBool>,
        _directory: TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = TempDir::new().unwrap();
            let manager = Arc::new(BorrowedSpillFixture::new(directory.path()).build().unwrap());
            let mut config = BufferManagerConfig::with_budget(1 << 20);
            config.soft_limit_fraction = 1.0;
            config.evict_limit_fraction = 1.0;
            config.hard_limit_fraction = 1.0;
            let memory = BufferManager::new(config);
            let hostile_capacity = Arc::new(AtomicBool::new(false));
            let capacity_flag = Arc::clone(&hostile_capacity);
            let state = PartitionedState::new_accounted_with_cancellation(
                Arc::clone(&manager),
                1,
                |_value: &Probe, _writer, _limits| {
                    panic!("resident drain must not serialize its aggregate value")
                },
                |_reader, _limits| {
                    panic!("resident drain must not deserialize its aggregate value")
                },
                move |value: &Probe| {
                    assert!(
                        !capacity_flag.load(Ordering::Acquire),
                        "hostile resident capacity callback"
                    );
                    Ok(value.bytes.capacity())
                },
                memory
                    .try_allocate(0, MemoryRegion::ExecutionBuffers)
                    .unwrap(),
                QueryExecutionControl::new().token(),
            )
            .unwrap();
            Self {
                state,
                memory,
                manager,
                dropped: Arc::new(AtomicUsize::new(0)),
                allocated_at_drop: Arc::new(AtomicUsize::new(0)),
                hostile_drop: Arc::new(AtomicBool::new(false)),
                hostile_capacity,
                _directory: directory,
            }
        }

        fn insert(&mut self, id: u8, bound: usize) {
            let memory = Arc::clone(&self.memory);
            let dropped = Arc::clone(&self.dropped);
            let allocated_at_drop = Arc::clone(&self.allocated_at_drop);
            let hostile_drop = Arc::clone(&self.hostile_drop);
            self.state
                .get_or_insert_with_accounted(vec![Value::Int64(i64::from(id))], bound, || Probe {
                    bytes: vec![id; 64],
                    memory,
                    dropped,
                    allocated_at_drop,
                    hostile_drop,
                })
                .unwrap();
        }
    }

    #[test]
    fn resident_cursor_moves_values_without_spilling_and_retires_buckets_at_eof() {
        let mut fixture = Fixture::new();
        fixture.insert(7, 128);
        fixture.insert(9, 128);
        let original = fixture.memory.allocated();
        let mut resident = ResidentPartition::begin(&mut fixture.state, 0);
        let map_bytes = resident.map_bytes;
        assert!(map_bytes > 0);
        assert_eq!(fixture.memory.allocated(), original);

        let first = resident.next(&mut fixture.state).unwrap().unwrap();
        let second = resident.next(&mut fixture.state).unwrap().unwrap();
        let ids = [first.value().bytes[0], second.value().bytes[0]];
        assert!(ids == [7, 9] || ids == [9, 7]);
        assert_eq!(first.key(), &[Value::Int64(i64::from(ids[0]))]);
        assert_eq!(second.key(), &[Value::Int64(i64::from(ids[1]))]);
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 0);
        let before_eof = fixture.memory.allocated();
        assert!(resident.next(&mut fixture.state).unwrap().is_none());
        assert_eq!(fixture.memory.allocated(), before_eof - map_bytes);
        assert!(resident.destroy());
        assert_eq!(fixture.manager.active_file_count(), 0);
        let (key, value) = first.into_accounted_parts();
        assert_eq!(value.granted_bytes(), 128);
        drop(key);
        let before_value_drop = fixture.memory.allocated();
        drop(value);
        assert_eq!(
            fixture.allocated_at_drop.load(Ordering::Acquire),
            before_value_drop
        );
        assert_eq!(fixture.memory.allocated(), before_value_drop - 128);
        drop(second);
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 2);
        assert_eq!(fixture.memory.allocated(), fixture.state.granted_bytes());
        assert_eq!(
            fixture.state.granted_bytes(),
            fixture.state.observed_resident_capacity_bytes().unwrap()
        );
    }

    #[test]
    fn resident_cursor_decode_admission_keeps_the_pending_value_under_root_authority() {
        let mut fixture = Fixture::new();
        fixture.insert(7, 64);
        let root_bytes = fixture.state.granted_bytes();
        let filler = fixture
            .memory
            .try_allocate((1 << 20) - root_bytes, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut resident = ResidentPartition::begin(&mut fixture.state, 0);
        let Err(error) = resident.next(&mut fixture.state) else {
            panic!("the decoded key requires unavailable memory");
        };
        assert!(error.resident_memory_error().is_some());
        assert!(resident.pending.is_some());
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 0);
        assert_eq!(fixture.state.granted_bytes(), root_bytes);
        assert!(resident.destroy());
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 1);
        assert_eq!(fixture.allocated_at_drop.load(Ordering::Acquire), 1 << 20);
        fixture.state.partition_sizes[0] = 0;
        fixture.state.reconcile_grant().unwrap();
        drop(filler);
        assert_eq!(fixture.memory.allocated(), fixture.state.granted_bytes());
        assert_eq!(fixture.manager.active_file_count(), 0);
    }

    #[test]
    fn resident_cursor_contains_multiple_hostile_drops_after_a_capacity_panic() {
        let mut fixture = Fixture::new();
        for id in 1..=3 {
            fixture.insert(id, 64);
        }
        let root_bytes = fixture.state.granted_bytes();
        let mut resident = ResidentPartition::begin(&mut fixture.state, 0);
        fixture.hostile_capacity.store(true, Ordering::Release);
        fixture.hostile_drop.store(true, Ordering::Release);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = resident.next(&mut fixture.state);
        }));
        assert!(panic.is_err());
        assert!(resident.pending.is_some());
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 0);
        assert!(!resident.destroy());
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 3);
        assert_eq!(
            fixture.allocated_at_drop.load(Ordering::Acquire),
            root_bytes
        );
        assert_eq!(fixture.memory.allocated(), root_bytes);
        assert_eq!(fixture.manager.active_file_count(), 0);
    }

    #[test]
    fn resident_cursor_rejects_missing_authority_before_entry_or_bucket_retirement() {
        let mut fixture = Fixture::new();
        fixture.insert(7, 64);
        let mut resident = ResidentPartition::begin(&mut fixture.state, 0);
        let root = fixture.state.grant.take();
        let result = resident.next(&mut fixture.state);
        fixture.state.grant = root;
        assert!(matches!(
            result,
            Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain has no root authority"
            })
        ));
        assert!(resident.pending.is_none());
        assert_eq!(
            resident.entries.as_ref().map(ExactSizeIterator::len),
            Some(1)
        );
        assert_eq!(fixture.dropped.load(Ordering::Acquire), 0);

        drop(resident.next(&mut fixture.state).unwrap().unwrap());
        let map_bytes = resident.map_bytes;
        let root = fixture.state.grant.as_mut().unwrap();
        // Keep the real charge live while testing the depleted root invariant.
        let held_authority = root.split(root.size()).unwrap();
        let result = resident.next(&mut fixture.state);
        fixture
            .state
            .grant
            .as_mut()
            .unwrap()
            .try_merge(held_authority)
            .unwrap();
        assert!(matches!(
            result,
            Err(PartitionOperationError::NativeMapInvariant {
                message: "resident drain map lost its root authority"
            })
        ));
        assert!(resident.entries.is_some());
        assert_eq!(resident.map_bytes, map_bytes);
        assert!(resident.next(&mut fixture.state).unwrap().is_none());
        assert!(resident.destroy());
        assert_eq!(fixture.memory.allocated(), fixture.state.granted_bytes());
    }
}
