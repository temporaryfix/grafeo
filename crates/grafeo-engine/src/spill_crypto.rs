//! Engine-owned root authentication and optional encryption for query spill.

mod authority;

use std::io;
use std::mem::size_of;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use grafeo_common::encryption::{ENCRYPTION_OVERHEAD, KeyChain, NONCE_SIZE, PageEncryptor};
use grafeo_common::types::StoreId;
use grafeo_common::utils::error::{Error, ErrorCode};
#[cfg(test)]
use grafeo_core::execution::spill::SpillManager;
use grafeo_core::execution::spill::{
    MAX_FIXED_CONTROL_PAYLOAD_BYTES, MAX_SPILL_RECORD_BYTES, NoopSpillIo, OpenSpillRecord,
    SpillDiskQuota, SpillFileIdentity, SpillFrameLimits, SpillQueryIdentity, SpillRecordMeta,
    SpillRecordProvider, SpillRoot, SpillRootAuthority,
};
use zeroize::Zeroize;

const FILE_KEY_CONTEXT: &str = "grafeo-spill-record-v1";
const ROOT_KEY_CONTEXT: &str = "grafeo-spill-root-auth-v1";
const ROOT_KEY_ID_CONTEXT: &str = "grafeo-spill-root-key-id-v1";
const ROOT_AUTH_MAGIC: [u8; 4] = *b"GSR1";

/// Shared database owner; lazy opening preserves resident-only startup behavior.
pub(crate) struct DatabaseSpillRoot {
    database_path: Option<std::path::PathBuf>,
    #[cfg(feature = "encryption")]
    encryption_key_chain: Option<Arc<KeyChain>>,
    max_spill_bytes: Option<u64>,
    max_root_spill_bytes: Option<u64>,
    opened: parking_lot::Mutex<OpenedSpillRoot>,
}

#[derive(Default)]
struct OpenedSpillRoot {
    ephemeral_key: Option<Arc<KeyChain>>,
    root: Option<(StoreId, Arc<SpillRoot>)>,
    #[cfg(test)]
    io: Option<Arc<dyn grafeo_core::execution::spill::SpillIo>>,
}

impl DatabaseSpillRoot {
    pub(crate) fn new(config: &crate::Config) -> Self {
        Self {
            database_path: config.path.clone(),
            #[cfg(feature = "encryption")]
            encryption_key_chain: config
                .encryption
                .as_ref()
                .map(|encryption| Arc::clone(&encryption.key_chain)),
            max_spill_bytes: config.max_query_spill_bytes,
            max_root_spill_bytes: config.max_root_spill_bytes,
            opened: parking_lot::Mutex::new(OpenedSpillRoot::default()),
        }
    }

    #[cfg(all(
        test,
        feature = "lpg",
        feature = "gql",
        feature = "async-storage",
        any(target_os = "linux", target_os = "macos")
    ))]
    pub(crate) fn install_test_io(&self, io: Arc<dyn grafeo_core::execution::spill::SpillIo>) {
        let mut opened = self.opened.lock();
        assert!(
            opened.root.is_none(),
            "install query faults before root admission"
        );
        opened.io = Some(io);
    }

    pub(crate) fn open(
        &self,
        path: &Path,
        store_id: StoreId,
        cancellation: &grafeo_core::execution::QueryCancellationToken,
    ) -> io::Result<Arc<SpillRoot>> {
        // Match the existing cooperative publication-lock protocol: each
        // bounded wait observes cancellation, without rejecting concurrent
        // first queries simply because another caller is initializing the root.
        let mut opened = loop {
            cancellation
                .check()
                .map_err(|reason| io::Error::new(io::ErrorKind::Interrupted, reason))?;
            if let Some(opened) = self
                .opened
                .try_lock_for(std::time::Duration::from_millis(1))
            {
                break opened;
            }
        };
        if let Some((bound_store, root)) = &opened.root
            && *bound_store == store_id
        {
            return Ok(Arc::clone(root));
        }
        #[cfg(feature = "encryption")]
        let encrypted_key = self.encryption_key_chain.as_ref();
        #[cfg(not(feature = "encryption"))]
        let encrypted_key: Option<&Arc<KeyChain>> = None;
        let key_chain = match encrypted_key {
            Some(key_chain) => Arc::clone(key_chain),
            None if self.database_path.is_none() => match &opened.ephemeral_key {
                Some(key) => Arc::clone(key),
                None => {
                    let key = authority::load_or_create_key_chain(None, store_id, path)?;
                    opened.ephemeral_key = Some(Arc::clone(&key));
                    key
                }
            },
            None => {
                authority::load_or_create_key_chain(self.database_path.as_deref(), store_id, path)?
            }
        };
        let key_id_secret = key_chain.derive_dek(ROOT_KEY_ID_CONTEXT, store_id.as_bytes());
        // A public identifier must not disclose the derived authentication key.
        let key_id = *blake3::hash(&*key_id_secret).as_bytes();
        let authority = Arc::new(EngineSpillRootAuthority {
            store_id,
            key_id,
            key_chain,
            encrypt_records: encrypted_key.is_some(),
        });
        let limits = if authority.encrypt_records {
            encrypted_frame_limits()?
        } else {
            SpillFrameLimits::format_max()
        };
        #[cfg(test)]
        let io = opened.io.clone().unwrap_or_else(|| Arc::new(NoopSpillIo));
        #[cfg(not(test))]
        let io = Arc::new(NoopSpillIo);
        let root = SpillRoot::open(
            path,
            authority,
            limits,
            io,
            SpillDiskQuota::new(self.max_spill_bytes.unwrap_or(u64::MAX)),
            self.max_root_spill_bytes,
        )?;
        root.scavenge()?;
        opened.root = Some((store_id, Arc::clone(&root)));
        Ok(root)
    }
}

struct EngineSpillRootAuthority {
    store_id: StoreId,
    key_id: [u8; 32],
    key_chain: Arc<KeyChain>,
    encrypt_records: bool,
}

impl SpillRootAuthority for EngineSpillRootAuthority {
    fn store_id(&self) -> StoreId {
        self.store_id
    }

    fn key_id(&self) -> [u8; 32] {
        self.key_id
    }

    fn authenticate_marker(&self, marker: &[u8]) -> io::Result<[u8; 32]> {
        // The complete bounded marker selects a domain-separated subkey and
        // nonce. Changing any generation/identity/AAD byte changes the subkey;
        // repeated authentication of identical bytes repeats identical empty
        // plaintext and AAD, never a different message under a reused nonce.
        if marker.len() > 512 {
            return Err(invalid_input(
                "spill authority marker exceeds its fixed bound",
            ));
        }
        let identity = blake3::hash(marker);
        let encryptor = self
            .key_chain
            .encryptor_for(ROOT_KEY_CONTEXT, identity.as_bytes());
        let mut nonce = [0; NONCE_SIZE];
        nonce.copy_from_slice(&identity.as_bytes()[..NONCE_SIZE]);
        let sealed = encryptor
            .encrypt(&[], &nonce, marker)
            .map_err(|error| crypto_io_error(error, io::ErrorKind::Other))?;
        let mut authenticator = [0; 32];
        authenticator[..4].copy_from_slice(&ROOT_AUTH_MAGIC);
        authenticator[4..].copy_from_slice(
            sealed
                .get(..28)
                .filter(|_| sealed.len() == 28)
                .ok_or_else(|| invalid_data("invalid spill root authenticator size"))?,
        );
        Ok(authenticator)
    }

    fn verify_marker(&self, marker: &[u8], authenticator: &[u8; 32]) -> io::Result<bool> {
        if marker.len() > 512 || authenticator[..4] != ROOT_AUTH_MAGIC {
            return Ok(false);
        }
        let identity = blake3::hash(marker);
        let encryptor = self
            .key_chain
            .encryptor_for(ROOT_KEY_CONTEXT, identity.as_bytes());
        // AES-GCM verifies the tag in constant time inside the existing provider.
        Ok(encryptor
            .decrypt(&authenticator[4..], marker)
            .is_ok_and(|plaintext| plaintext.is_empty()))
    }

    fn record_provider(
        &self,
        identity: SpillQueryIdentity,
    ) -> io::Result<Arc<dyn SpillRecordProvider>> {
        if self.encrypt_records {
            let provider = AuthenticatedSpillRecordProvider::new(Arc::clone(&self.key_chain));
            provider.bind_query(identity)?;
            Ok(Arc::new(provider))
        } else {
            Ok(Arc::new(
                grafeo_core::execution::spill::CleartextSpillRecordProvider,
            ))
        }
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn crypto_io_error(error: Error, fallback: io::ErrorKind) -> io::Error {
    let kind = if error.error_code() == ErrorCode::StorageFull {
        io::ErrorKind::OutOfMemory
    } else {
        fallback
    };
    io::Error::from(kind)
}

fn file_key_id(query: SpillQueryIdentity, file: SpillFileIdentity) -> [u8; 32] {
    let mut id = [0u8; 32];
    id[..16].copy_from_slice(query.as_bytes());
    id[16..].copy_from_slice(file.as_bytes());
    id
}

fn record_nonce(identity: SpillFileIdentity, sequence: u64) -> [u8; NONCE_SIZE] {
    let mut nonce = [0u8; NONCE_SIZE];
    nonce.copy_from_slice(&identity.as_bytes()[..NONCE_SIZE]);
    for (destination, counter) in nonce[NONCE_SIZE - 8..]
        .iter_mut()
        .zip(sequence.to_be_bytes())
    {
        *destination ^= counter;
    }
    nonce
}

/// Per-query provider whose key bytes never cross into `grafeo-core`.
pub(crate) struct AuthenticatedSpillRecordProvider {
    key_chain: Arc<KeyChain>,
    query_identity: OnceLock<SpillQueryIdentity>,
}

impl AuthenticatedSpillRecordProvider {
    pub(crate) fn new(key_chain: Arc<KeyChain>) -> Self {
        Self {
            key_chain,
            query_identity: OnceLock::new(),
        }
    }

    fn bind_query(&self, identity: SpillQueryIdentity) -> io::Result<()> {
        if let Some(bound) = self.query_identity.get() {
            if *bound == identity {
                return Ok(());
            }
            return Err(invalid_input(
                "authenticated spill provider cannot be reused across query identities",
            ));
        }
        self.query_identity
            .set(identity)
            .map_err(|_| invalid_input("authenticated spill provider query binding raced"))
    }
}

impl SpillRecordProvider for AuthenticatedSpillRecordProvider {
    fn seals(&self) -> bool {
        true
    }

    fn begin_file(&self, identity: SpillFileIdentity) -> io::Result<Box<dyn OpenSpillRecord>> {
        let query_identity = *self.query_identity.get().ok_or_else(|| {
            invalid_input("authenticated spill provider is not bound to a query identity")
        })?;
        let encryptor = self
            .key_chain
            .encryptor_for(FILE_KEY_CONTEXT, &file_key_id(query_identity, identity));
        Ok(Box::new(AuthenticatedOpenSpillRecord {
            identity,
            encryptor,
            next_sequence: 0,
        }))
    }

    fn file_workspace_allocation_bound(&self) -> Option<usize> {
        let control_stored_len =
            MAX_FIXED_CONTROL_PAYLOAD_BYTES.checked_add(ENCRYPTION_OVERHEAD)?;
        let control_peak =
            PageEncryptor::encrypt_allocation_bound(MAX_FIXED_CONTROL_PAYLOAD_BYTES)?
                .max(PageEncryptor::decrypt_allocation_bound(control_stored_len)?);
        let retained_state_bound = size_of::<AuthenticatedOpenSpillRecord>().checked_mul(2)?;
        retained_state_bound.checked_add(control_peak)
    }

    fn supports_qualified_exact_open(&self) -> bool {
        true
    }
}

struct AuthenticatedOpenSpillRecord {
    identity: SpillFileIdentity,
    encryptor: PageEncryptor,
    next_sequence: u64,
}

impl AuthenticatedOpenSpillRecord {
    fn validate_meta(&self, meta: &SpillRecordMeta) -> io::Result<()> {
        if !meta.is_sealed() {
            return Err(invalid_data(
                "authenticated spill record is not marked sealed",
            ));
        }
        if meta.identity() != self.identity {
            return Err(invalid_data("authenticated spill file identity changed"));
        }
        if meta.sequence() != self.next_sequence {
            return Err(invalid_data(format!(
                "authenticated spill sequence {} does not match expected {}",
                meta.sequence(),
                self.next_sequence
            )));
        }
        Ok(())
    }

    fn next_sequence(&self) -> io::Result<u64> {
        self.next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("authenticated spill record counter exhausted"))
    }

    fn validate_meta_qualified(&self, meta: &SpillRecordMeta) -> io::Result<()> {
        if !meta.is_sealed()
            || meta.identity() != self.identity
            || meta.sequence() != self.next_sequence
        {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
        Ok(())
    }

    fn next_sequence_qualified(&self) -> io::Result<u64> {
        self.next_sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
    }

    fn open_sequence_qualified_into(
        &mut self,
        sequence: u64,
        aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> io::Result<()> {
        let result = (|| {
            if sequence != self.next_sequence {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
            let next_sequence = self.next_sequence_qualified()?;
            let expected_nonce = record_nonce(self.identity, sequence);
            if stored.get(..NONCE_SIZE) != Some(expected_nonce.as_slice()) {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
            self.encryptor
                .decrypt_into(stored, aad, plaintext)
                .map_err(|error| crypto_io_error(error, io::ErrorKind::InvalidData))?;
            self.next_sequence = next_sequence;
            Ok(())
        })();
        if result.is_err() {
            plaintext.zeroize();
        }
        result
    }
}

impl OpenSpillRecord for AuthenticatedOpenSpillRecord {
    fn stored_len(&self, plaintext_len: usize) -> io::Result<usize> {
        plaintext_len
            .checked_add(ENCRYPTION_OVERHEAD)
            .ok_or_else(|| invalid_input("authenticated spill record length overflow"))
    }

    fn seal_allocation_bound(&self, plaintext_len: usize) -> Option<usize> {
        PageEncryptor::encrypt_allocation_bound(plaintext_len)
    }

    fn open_allocation_bound(&self, stored_len: usize) -> Option<usize> {
        PageEncryptor::decrypt_allocation_bound(stored_len)
    }

    fn seal(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        plaintext: &[u8],
    ) -> io::Result<Vec<u8>> {
        self.validate_meta(meta)?;
        let next_sequence = self.next_sequence()?;
        if meta.plaintext_len() != u64::try_from(plaintext.len()).unwrap_or(u64::MAX) {
            return Err(invalid_data("authenticated spill plaintext length changed"));
        }
        let expected_stored = self.stored_len(plaintext.len())?;
        if meta.stored_len() != u64::try_from(expected_stored).unwrap_or(u64::MAX) {
            return Err(invalid_data("authenticated spill stored length changed"));
        }
        let nonce = record_nonce(self.identity, meta.sequence());
        let sealed = self
            .encryptor
            .encrypt(plaintext, &nonce, aad)
            .map_err(|error| crypto_io_error(error, io::ErrorKind::Other))?;
        if sealed.len() != expected_stored {
            return Err(invalid_data("authenticated spill provider length drift"));
        }
        self.next_sequence = next_sequence;
        Ok(sealed)
    }

    fn open(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        stored: &[u8],
    ) -> io::Result<Vec<u8>> {
        self.validate_meta(meta)?;
        let next_sequence = self.next_sequence()?;
        let plaintext_len = usize::try_from(meta.plaintext_len())
            .map_err(|_| invalid_data("authenticated spill plaintext length is not addressable"))?;
        if stored.len() != self.stored_len(plaintext_len)? {
            return Err(invalid_data(
                "authenticated spill ciphertext length changed",
            ));
        }
        let expected_nonce = record_nonce(self.identity, meta.sequence());
        if stored.get(..NONCE_SIZE) != Some(expected_nonce.as_slice()) {
            return Err(invalid_data(
                "authenticated spill nonce does not match its file counter",
            ));
        }
        let plaintext = self
            .encryptor
            .decrypt(stored, aad)
            .map_err(|error| crypto_io_error(error, io::ErrorKind::InvalidData))?;
        if plaintext.len() != plaintext_len {
            return Err(invalid_data("authenticated spill plaintext length drift"));
        }
        self.next_sequence = next_sequence;
        Ok(plaintext)
    }

    fn open_qualified_into(
        &mut self,
        meta: &SpillRecordMeta,
        aad: &[u8; 32],
        stored: &[u8],
        plaintext: &mut [u8],
    ) -> Option<io::Result<()>> {
        let result = (|| {
            self.validate_meta_qualified(meta)?;
            let plaintext_len = usize::try_from(meta.plaintext_len())
                .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
            if plaintext.len() != plaintext_len {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
            let expected_stored = plaintext_len
                .checked_add(ENCRYPTION_OVERHEAD)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            if stored.len() != expected_stored {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
            if meta.stored_len() != u64::try_from(stored.len()).unwrap_or(u64::MAX) {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
            self.open_sequence_qualified_into(meta.sequence(), aad, stored, plaintext)
        })();
        if result.is_err() {
            plaintext.zeroize();
        }
        Some(result)
    }
}

fn encrypted_frame_limits() -> io::Result<SpillFrameLimits> {
    let format_max = usize::try_from(MAX_SPILL_RECORD_BYTES)
        .map_err(|_| invalid_input("spill format maximum is not addressable"))?;
    let max_plaintext = format_max
        .checked_sub(ENCRYPTION_OVERHEAD)
        .ok_or_else(|| invalid_input("spill encryption overhead exceeds the format maximum"))?;
    SpillFrameLimits::new(max_plaintext, format_max)
}

// These operator fixtures inject file failures only after real root/query
// admission. Root-admission fault coverage lives in the spill-root tests.
#[cfg(test)]
pub(crate) fn admitted_spill_test_resources(
    parent: &Path,
    memory: Arc<grafeo_common::memory::buffer::BufferManager>,
    cancellation: grafeo_core::execution::QueryCancellationToken,
    provider: Arc<dyn SpillRecordProvider>,
    limits: SpillFrameLimits,
    io: Arc<dyn grafeo_core::execution::spill::SpillIo>,
    quota: SpillDiskQuota,
) -> (
    grafeo_core::execution::QueryResourceContext,
    Arc<SpillManager>,
) {
    use grafeo_core::execution::spill::{SpillIo, SpillIoOperation};
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Authority {
        owner: EngineSpillRootAuthority,
        records: Arc<dyn SpillRecordProvider>,
    }
    impl SpillRootAuthority for Authority {
        fn store_id(&self) -> StoreId {
            self.owner.store_id()
        }
        fn key_id(&self) -> [u8; 32] {
            self.owner.key_id()
        }
        fn authenticate_marker(&self, bytes: &[u8]) -> io::Result<[u8; 32]> {
            self.owner.authenticate_marker(bytes)
        }
        fn verify_marker(&self, bytes: &[u8], seal: &[u8; 32]) -> io::Result<bool> {
            self.owner.verify_marker(bytes, seal)
        }
        fn record_provider(
            &self,
            _: SpillQueryIdentity,
        ) -> io::Result<Arc<dyn SpillRecordProvider>> {
            Ok(Arc::clone(&self.records))
        }
    }
    struct FileHooks {
        admitted: AtomicBool,
        inner: Arc<dyn SpillIo>,
    }
    impl SpillIo for FileHooks {
        fn check(&self, operation: SpillIoOperation) -> io::Result<()> {
            if self.admitted.load(Ordering::SeqCst) {
                self.inner.check(operation)
            } else {
                Ok(())
            }
        }
        fn qualified_sort_hook_workspace_bound(&self) -> Option<usize> {
            self.inner.qualified_sort_hook_workspace_bound()
        }
        fn qualified_reader_hook_workspace_bound(&self) -> Option<usize> {
            self.inner.qualified_reader_hook_workspace_bound()
        }
    }
    let hooks = Arc::new(FileHooks {
        admitted: AtomicBool::new(false),
        inner: io,
    });
    let authority = Arc::new(Authority {
        owner: EngineSpillRootAuthority {
            store_id: StoreId::from_bytes([0x25; 32]).unwrap(),
            key_id: [0x45; 32],
            key_chain: Arc::new(KeyChain::new([0x35; 32])),
            encrypt_records: false,
        },
        records: provider,
    });
    let root = SpillRoot::open(
        parent,
        authority,
        limits,
        hooks.clone(),
        quota,
        Some(64 << 20),
    )
    .unwrap();
    let resources =
        grafeo_core::execution::QueryResourceContext::with_spill_root(memory, &root, cancellation)
            .unwrap();
    let manager = Arc::clone(resources.ensure_spill_manager().unwrap().unwrap());
    hooks.admitted.store(true, Ordering::SeqCst);
    (resources, manager)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::HashSet;
    use std::sync::Arc;

    use grafeo_common::encryption::{KeyChain, NONCE_SIZE, PageEncryptor};
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::{Error, StorageError};
    use grafeo_core::execution::spill::{
        ExternalSort, SemanticRowComparator, SpillFileRole, SpillManager, SpillQueryIdentity,
        SpillRecordProvider, SpillRoot, SpillRootAuthority,
    };
    use grafeo_core::execution::{QueryExecutionControl, QueryResourceContext};

    use super::{
        AuthenticatedOpenSpillRecord, AuthenticatedSpillRecordProvider, EngineSpillRootAuthority,
        crypto_io_error, record_nonce,
    };

    struct AuthenticatedTestAuthority {
        owner: EngineSpillRootAuthority,
        provider: Arc<AuthenticatedSpillRecordProvider>,
    }

    impl SpillRootAuthority for AuthenticatedTestAuthority {
        fn store_id(&self) -> grafeo_common::types::StoreId {
            self.owner.store_id()
        }
        fn key_id(&self) -> [u8; 32] {
            self.owner.key_id()
        }
        fn authenticate_marker(&self, marker: &[u8]) -> std::io::Result<[u8; 32]> {
            self.owner.authenticate_marker(marker)
        }
        fn verify_marker(&self, marker: &[u8], seal: &[u8; 32]) -> std::io::Result<bool> {
            self.owner.verify_marker(marker, seal)
        }
        fn record_provider(
            &self,
            identity: SpillQueryIdentity,
        ) -> std::io::Result<Arc<dyn SpillRecordProvider>> {
            self.provider.bind_query(identity)?;
            Ok(self.provider.clone())
        }
    }

    fn authenticated_test_root(
        parent: &std::path::Path,
        key_chain: Arc<KeyChain>,
        max_spill_bytes: Option<u64>,
    ) -> std::io::Result<(Arc<SpillRoot>, Arc<AuthenticatedSpillRecordProvider>)> {
        let provider = Arc::new(AuthenticatedSpillRecordProvider::new(Arc::clone(
            &key_chain,
        )));
        let authority = Arc::new(AuthenticatedTestAuthority {
            owner: EngineSpillRootAuthority {
                store_id: grafeo_common::types::StoreId::from_bytes([0x26; 32]).unwrap(),
                key_id: [0x46; 32],
                key_chain,
                encrypt_records: true,
            },
            provider: Arc::clone(&provider),
        });
        let root = SpillRoot::open(
            parent,
            authority,
            super::encrypted_frame_limits()?,
            Arc::new(super::NoopSpillIo),
            super::SpillDiskQuota::new(max_spill_bytes.unwrap_or(u64::MAX)),
            Some(64 << 20),
        )?;
        Ok((root, provider))
    }

    fn authenticated_test_resources(
        parent: &std::path::Path,
        key_chain: Arc<KeyChain>,
        max_spill_bytes: Option<u64>,
    ) -> (QueryResourceContext, Arc<SpillManager>) {
        let (root, _) = authenticated_test_root(parent, key_chain, max_spill_bytes).unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(16 << 20),
            &root,
            QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = Arc::clone(resources.ensure_spill_manager().unwrap().unwrap());
        (resources, manager)
    }

    #[cfg(all(
        feature = "grafeo-file",
        feature = "lpg",
        feature = "gql",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[test]
    fn scavenge_persistent_engine_child() {
        let Some(parent) = std::env::var_os("GRAFEO_ENGINE_SCAVENGE_PARENT") else {
            return;
        };
        let parent = std::path::Path::new(&parent);
        let path = parent.join("store.grafeo");
        let spill = parent.join("spill");
        let mut config = crate::Config::persistent(&path).with_spill_path(&spill);
        config.wal_enabled = false;
        #[cfg(feature = "encryption")]
        if std::env::var_os("GRAFEO_ENGINE_SCAVENGE_ENCRYPTED").is_some() {
            config.encryption = Some(crate::config::EncryptionConfig {
                key_chain: Arc::new(KeyChain::new([0x3d; 32])),
            });
        }
        let database = crate::GrafeoDB::with_config(config.clone()).unwrap();
        let store = database.store_id();
        database.close().unwrap();
        let owner = super::DatabaseSpillRoot::new(&config);
        let control = QueryExecutionControl::new();
        let root = owner.open(&spill, store, &control.token()).unwrap();
        let context = QueryResourceContext::new_with_cancellation(
            BufferManager::with_budget(1 << 20),
            control.token(),
        )
        .unwrap();
        let manager = SpillManager::from_query_lease(
            root.begin_query(context.query_id(), control.token())
                .unwrap(),
        );
        let _file = write_one_row(&manager, b"persisted crash spill");
        std::fs::write(
            parent.join("dead-leaf"),
            manager.spill_dir().as_os_str().as_encoded_bytes(),
        )
        .unwrap();
        std::process::exit(0);
    }

    #[cfg(all(
        feature = "grafeo-file",
        feature = "lpg",
        feature = "gql",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[test]
    fn scavenge_persistent_restart_runs_through_public_query_caller() {
        for encrypted in [false, true] {
            if encrypted && !cfg!(feature = "encryption") {
                continue;
            }
            let parent = tempfile::tempdir().unwrap();
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "spill_crypto::tests::scavenge_persistent_engine_child",
                    "--nocapture",
                ])
                .env("GRAFEO_ENGINE_SCAVENGE_PARENT", parent.path());
            if encrypted {
                command.env("GRAFEO_ENGINE_SCAVENGE_ENCRYPTED", "1");
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let leaf = std::path::PathBuf::from(
                std::fs::read_to_string(parent.path().join("dead-leaf")).unwrap(),
            );
            assert!(leaf.is_dir());
            let mut config = crate::Config::persistent(parent.path().join("store.grafeo"))
                .with_spill_path(parent.path().join("spill"));
            config.wal_enabled = false;
            #[cfg(feature = "encryption")]
            if encrypted {
                config.encryption = Some(crate::config::EncryptionConfig {
                    key_chain: Arc::new(KeyChain::new([0x3d; 32])),
                });
            }
            let database = crate::GrafeoDB::with_config(config).unwrap();
            let profile = database.execute("PROFILE RETURN 1").unwrap();
            let grafeo_common::types::Value::String(text) = &profile.rows()[0][0] else {
                panic!("PROFILE text");
            };
            assert!(
                text.contains("root-wide last_cleanup_inspected=1 removed=1"),
                "{text}"
            );
            assert!(
                text.contains("reserved_bytes_at_pass=1048576 truncated=false"),
                "{text}"
            );
            drop(profile);
            assert!(
                !leaf.exists(),
                "public query must trigger authenticated restart cleanup"
            );
            let ledger = std::fs::read(leaf.parent().unwrap().join(".grafeo-spill-quota")).unwrap();
            assert_eq!(
                u64::from_le_bytes(ledger[64..72].try_into().unwrap()),
                1 << 20
            );
            database.close().unwrap();
        }
    }

    const RECORD_HEADER_BYTES: usize = 36;

    #[test]
    fn root_authentication_binds_every_marker_byte_and_rejects_foreign_keys() {
        let store_id = grafeo_common::types::StoreId::from_bytes([3; 32]).unwrap();
        let provider = super::EngineSpillRootAuthority {
            store_id,
            key_id: [9; 32],
            key_chain: Arc::new(KeyChain::new([5; 32])),
            encrypt_records: false,
        };
        let foreign = super::EngineSpillRootAuthority {
            store_id,
            key_id: [9; 32],
            key_chain: Arc::new(KeyChain::new([6; 32])),
            encrypt_records: false,
        };
        for length in [109, 133, 149] {
            let marker = vec![7; length];
            let auth = provider.authenticate_marker(&marker).unwrap();
            assert!(provider.verify_marker(&marker, &auth).unwrap());
            assert!(!foreign.verify_marker(&marker, &auth).unwrap());
            assert_eq!(provider.authenticate_marker(&marker).unwrap(), auth);
            for index in 0..length {
                let mut changed = marker.clone();
                changed[index] ^= 1;
                assert!(!provider.verify_marker(&changed, &auth).unwrap());
                let changed_auth = provider.authenticate_marker(&changed).unwrap();
                assert_ne!(changed_auth, auth);
                assert!(provider.verify_marker(&changed, &changed_auth).unwrap());
            }
        }
        assert!(provider.authenticate_marker(&[0; 513]).is_err());
    }

    fn record_offsets(bytes: &[u8]) -> Vec<usize> {
        let mut offsets = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            assert!(bytes.len() - offset >= RECORD_HEADER_BYTES);
            offsets.push(offset);
            let stored_len = u64::from_le_bytes(
                bytes[offset + 24..offset + 32]
                    .try_into()
                    .expect("fixed stored-length field"),
            );
            let stored_len = usize::try_from(stored_len).expect("test record is addressable");
            offset = offset
                .checked_add(RECORD_HEADER_BYTES)
                .and_then(|next| next.checked_add(stored_len))
                .expect("test record offset does not overflow");
            assert!(offset <= bytes.len());
        }
        offsets
    }

    fn rewrite_record_crc(bytes: &mut [u8], record_offset: usize) {
        let stored_len = u64::from_le_bytes(
            bytes[record_offset + 24..record_offset + 32]
                .try_into()
                .expect("fixed stored-length field"),
        );
        let stored_len = usize::try_from(stored_len).expect("test record is addressable");
        let payload_start = record_offset + RECORD_HEADER_BYTES;
        let mut checksum = crc32fast::Hasher::new();
        checksum.update(&bytes[record_offset..record_offset + 32]);
        checksum.update(&bytes[payload_start..payload_start + stored_len]);
        bytes[record_offset + 32..record_offset + RECORD_HEADER_BYTES]
            .copy_from_slice(&checksum.finalize().to_le_bytes());
    }

    fn write_one_row(manager: &SpillManager, row: &[u8]) -> grafeo_core::execution::SpillFile {
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 1).unwrap();
        file.write_sort_row(row).unwrap();
        file.finish_write().unwrap();
        file
    }

    #[test]
    fn root_marker_requires_the_database_key_and_exact_message() {
        let authority = |key| EngineSpillRootAuthority {
            store_id: grafeo_common::types::StoreId::from_bytes([3; 32]).unwrap(),
            key_id: [4; 32],
            key_chain: Arc::new(KeyChain::new([key; 32])),
            encrypt_records: true,
        };
        let correct = authority(0x31);
        let wrong = authority(0x32);
        let marker = b"bounded query identity and retained inode";
        let seal = correct.authenticate_marker(marker).unwrap();
        assert!(correct.verify_marker(marker, &seal).unwrap());
        assert!(!wrong.verify_marker(marker, &seal).unwrap());
        let mut changed = marker.to_vec();
        changed[0] ^= 1;
        assert!(!correct.verify_marker(&changed, &seal).unwrap());
        let mut changed_seal = seal;
        changed_seal[31] ^= 1;
        assert!(!correct.verify_marker(marker, &changed_seal).unwrap());
    }

    #[test]
    fn provider_is_bound_to_exactly_one_query_identity() {
        let directory = tempfile::tempdir().unwrap();
        let (root, provider) =
            authenticated_test_root(directory.path(), Arc::new(KeyChain::new([0x41; 32])), None)
                .unwrap();
        let first = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(1 << 20),
            &root,
            QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = first.ensure_spill_manager().unwrap().unwrap();
        let bound = *provider.query_identity.get().unwrap();
        provider.bind_query(bound).unwrap();
        let second = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(1 << 20),
            &root,
            QueryExecutionControl::new().token(),
        )
        .unwrap();
        let error = second
            .ensure_spill_manager()
            .expect_err("provider reuse across query identities must fail closed");
        assert!(matches!(
            error,
            grafeo_core::execution::QueryResourceContextError::SpillAdmission {
                kind: std::io::ErrorKind::InvalidInput,
                ..
            }
        ));
        assert!(
            error
                .to_string()
                .contains("cannot be reused across query identities")
        );
        assert!(second.spill_manager().is_none());
        assert_eq!(*provider.query_identity.get().unwrap(), bound);
        assert!(manager.spill_dir().is_dir());
    }

    #[test]
    fn provider_declares_qualified_workspace_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let (root, provider) =
            authenticated_test_root(directory.path(), Arc::new(KeyChain::new([0x42; 32])), None)
                .unwrap();
        let resources = QueryResourceContext::with_spill_root(
            BufferManager::with_budget(1 << 20),
            &root,
            QueryExecutionControl::new().token(),
        )
        .unwrap();
        let manager = resources.ensure_spill_manager().unwrap().unwrap();
        let file = manager.create_file(SpillFileRole::SortRun).unwrap();
        let open_record = provider.begin_file(file.identity()).unwrap();

        assert!(provider.file_workspace_allocation_bound().is_some());
        assert!(provider.supports_qualified_exact_open());
        assert!(open_record.seal_allocation_bound(1024).is_some());
        let stored_len = open_record.stored_len(1024).unwrap();
        assert!(open_record.open_allocation_bound(stored_len).is_some());
    }

    #[test]
    fn authenticated_exact_open_round_trips_and_advances_only_after_success() {
        let directory = tempfile::tempdir().unwrap();
        let (_resources, manager) = authenticated_test_resources(
            directory.path(),
            Arc::new(KeyChain::new([0x44; 32])),
            None,
        );
        let identity = manager
            .create_file(SpillFileRole::SortRun)
            .unwrap()
            .identity();
        let key = [0xA4; 32];
        let aad = [0xAD; 32];
        let plaintext = b"authenticated exact spill output";
        let encrypted = PageEncryptor::new(&key)
            .encrypt(plaintext, &record_nonce(identity, 0), &aad)
            .unwrap();
        let mut open_record = AuthenticatedOpenSpillRecord {
            identity,
            encryptor: PageEncryptor::new(&key),
            next_sequence: 0,
        };

        let mut rejected = vec![0xA5; plaintext.len()];
        open_record
            .open_sequence_qualified_into(0, &[0xEE; 32], &encrypted, &mut rejected)
            .unwrap_err();
        assert_eq!(rejected, vec![0; plaintext.len()]);
        assert_eq!(open_record.next_sequence, 0);

        let mut opened = vec![0xA5; plaintext.len()];
        open_record
            .open_sequence_qualified_into(0, &aad, &encrypted, &mut opened)
            .unwrap();
        assert_eq!(opened, plaintext);
        assert_eq!(open_record.next_sequence, 1);

        let mut stale = vec![0xA5; plaintext.len()];
        open_record
            .open_sequence_qualified_into(0, &aad, &encrypted, &mut stale)
            .unwrap_err();
        assert_eq!(stale, vec![0; plaintext.len()]);
        assert_eq!(open_record.next_sequence, 1);
    }

    #[test]
    fn authenticated_exact_open_rejects_length_and_tamper_without_consuming_sequence() {
        let directory = tempfile::tempdir().unwrap();
        let (_resources, manager) = authenticated_test_resources(
            directory.path(),
            Arc::new(KeyChain::new([0x45; 32])),
            None,
        );
        let identity = manager
            .create_file(SpillFileRole::SortRun)
            .unwrap()
            .identity();
        let key = [0xA5; 32];
        let aad = [0xAC; 32];
        let plaintext = b"exact sequence retry";
        let encrypted = PageEncryptor::new(&key)
            .encrypt(plaintext, &record_nonce(identity, 0), &aad)
            .unwrap();
        let mut open_record = AuthenticatedOpenSpillRecord {
            identity,
            encryptor: PageEncryptor::new(&key),
            next_sequence: 0,
        };

        let mut wrong_length = vec![0xA5; plaintext.len() - 1];
        open_record
            .open_sequence_qualified_into(0, &aad, &encrypted, &mut wrong_length)
            .unwrap_err();
        assert_eq!(wrong_length, vec![0; plaintext.len() - 1]);
        assert_eq!(open_record.next_sequence, 0);

        let mut tampered = encrypted.clone();
        tampered[NONCE_SIZE] ^= 0x20;
        let mut rejected = vec![0xA5; plaintext.len()];
        open_record
            .open_sequence_qualified_into(0, &aad, &tampered, &mut rejected)
            .unwrap_err();
        assert_eq!(rejected, vec![0; plaintext.len()]);
        assert_eq!(open_record.next_sequence, 0);

        let mut opened = vec![0xA5; plaintext.len()];
        open_record
            .open_sequence_qualified_into(0, &aad, &encrypted, &mut opened)
            .unwrap();
        assert_eq!(opened, plaintext);
        assert_eq!(open_record.next_sequence, 1);
    }

    #[test]
    fn crypto_allocation_failure_remains_structured_for_spill() {
        let error = crypto_io_error(
            Error::Storage(StorageError::Full),
            std::io::ErrorKind::Other,
        );
        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
    }

    #[test]
    fn encrypted_provider_executes_qualified_external_sort() {
        let directory = tempfile::tempdir().unwrap();
        let (resources, manager) = authenticated_test_resources(
            directory.path(),
            Arc::new(KeyChain::new([0x43; 32])),
            None,
        );
        let buffer_manager = Arc::clone(resources.buffer_manager());
        let grant = resources.try_allocate(0).unwrap();
        let comparator = SemanticRowComparator::new(|left, right| match (&left[0], &right[0]) {
            (Value::Int64(left), Value::Int64(right)) => left.cmp(right),
            _ => Ordering::Equal,
        });
        let mut sort = ExternalSort::new_accounted_with_comparator_and_cancellation(
            Arc::clone(&manager),
            1,
            comparator,
            grant,
            resources.cancellation_token().clone(),
        );
        sort.spill_sorted_run(vec![vec![Value::Int64(1)], vec![Value::Int64(3)]])
            .unwrap();
        sort.spill_sorted_run(vec![vec![Value::Int64(2)], vec![Value::Int64(4)]])
            .unwrap();

        let mut observed = Vec::new();
        let mut cursor = sort.merge_cursor(Vec::new(), 1).unwrap();
        while let Some(chunk) = cursor.next_chunk().unwrap() {
            for row in chunk.rows() {
                let Value::Int64(value) = row[0] else {
                    panic!("qualified encrypted sort changed the row type");
                };
                observed.push(value);
            }
        }
        drop(cursor);

        assert_eq!(observed, [1, 2, 3, 4]);
        assert_eq!(manager.active_file_count(), 0);
        drop(sort);
        drop(resources);
        assert_eq!(buffer_manager.allocated(), 0);
    }

    #[test]
    fn record_nonces_are_counter_bound_and_distinct_between_files() {
        let directory = tempfile::tempdir().unwrap();
        let (_resources, manager) = authenticated_test_resources(
            directory.path(),
            Arc::new(KeyChain::new([0x51; 32])),
            None,
        );
        let first = write_one_row(&manager, b"first");
        let second = write_one_row(&manager, b"second");
        assert_ne!(first.identity(), second.identity());

        let mut observed = HashSet::new();
        for file in [&first, &second] {
            let bytes = std::fs::read(file.path()).unwrap();
            for (sequence, offset) in record_offsets(&bytes).into_iter().enumerate() {
                let payload = &bytes[offset + RECORD_HEADER_BYTES..];
                let nonce: [u8; NONCE_SIZE] = payload[..NONCE_SIZE]
                    .try_into()
                    .expect("sealed record nonce");
                assert_eq!(nonce, record_nonce(file.identity(), sequence as u64));
                assert!(observed.insert(nonce), "nonce reuse across spill records");
            }
        }
    }

    #[test]
    fn record_nonces_do_not_repeat_across_query_recreation() {
        let directory = tempfile::tempdir().unwrap();
        let key_chain = Arc::new(KeyChain::new([0x52; 32]));

        let (first_nonce, first_leaf) = {
            let (_resources, manager) =
                authenticated_test_resources(directory.path(), Arc::clone(&key_chain), None);
            let first_leaf = manager.spill_dir().to_path_buf();
            let file = write_one_row(&manager, b"before restart");
            let bytes = std::fs::read(file.path()).unwrap();
            let offset = record_offsets(&bytes)[0];
            let nonce = <[u8; NONCE_SIZE]>::try_from(
                &bytes[offset + RECORD_HEADER_BYTES..offset + RECORD_HEADER_BYTES + NONCE_SIZE],
            )
            .unwrap();
            (nonce, first_leaf)
        };
        assert!(!first_leaf.exists(), "finished query leaf must be removed");
        let mut retained: Vec<_> = std::fs::read_dir(first_leaf.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        retained.sort();
        assert_eq!(
            retained,
            [
                ".grafeo-spill-quota",
                ".grafeo-spill-quota.lock",
                ".grafeo-spill-root"
            ]
            .map(std::ffi::OsString::from)
        );

        let (_resources, manager) = authenticated_test_resources(directory.path(), key_chain, None);
        let file = write_one_row(&manager, b"after restart");
        let bytes = std::fs::read(file.path()).unwrap();
        let offset = record_offsets(&bytes)[0];
        let second_nonce = <[u8; NONCE_SIZE]>::try_from(
            &bytes[offset + RECORD_HEADER_BYTES..offset + RECORD_HEADER_BYTES + NONCE_SIZE],
        )
        .unwrap();
        assert_ne!(first_nonce, second_nonce);
    }

    #[test]
    fn ciphertext_tampering_fails_authentication_after_valid_crc() {
        let directory = tempfile::tempdir().unwrap();
        let (_resources, manager) = authenticated_test_resources(
            directory.path(),
            Arc::new(KeyChain::new([0x61; 32])),
            None,
        );
        let file = write_one_row(&manager, b"authenticated row");
        let mut bytes = std::fs::read(file.path()).unwrap();
        let offsets = record_offsets(&bytes);
        let row_offset = offsets[2];
        bytes[row_offset + RECORD_HEADER_BYTES + NONCE_SIZE] ^= 0x80;
        rewrite_record_crc(&mut bytes, row_offset);
        std::fs::write(file.path(), bytes).unwrap();

        let mut reader = file.reader().unwrap();
        assert_eq!(reader.read_sort_run_start().unwrap(), (1, 1));
        let error = reader.read_sort_row().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
}
