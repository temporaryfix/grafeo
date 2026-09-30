//! Encryption at rest primitives for Grafeo.
//!
//! Provides AES-256-GCM authenticated encryption with a hierarchical key system:
//!
//! - **Root key**: derived from a password (Argon2id) or provided externally
//! - **Master encryption key (ME)**: randomly generated, wrapped (encrypted) by the root key
//! - **Data encryption keys (DEKs)**: derived deterministically from the ME via HKDF
//!
//! Each storage component (WAL, snapshots, vector pages, spill files) gets its own DEK
//! derived from a unique context string and component ID. Nonces are counter-based
//! (no randomness needed) because each component has a natural monotonic counter.
//!
//! # Feature flag
//!
//! This module requires the `encryption` feature. When disabled, no encryption code
//! is compiled and the database operates with zero overhead.

use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce, Tag};
use hkdf::Hkdf;
use rand::RngExt;
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

use crate::utils::error::{CryptoError, Error, Result, StorageError};

/// Size of AES-256-GCM nonce in bytes.
pub const NONCE_SIZE: usize = 12;

/// Size of AES-256-GCM authentication tag in bytes.
pub const TAG_SIZE: usize = 16;

/// Size of encryption keys in bytes (256-bit).
pub const KEY_SIZE: usize = 32;

/// Overhead added per encrypted record: nonce + tag.
pub const ENCRYPTION_OVERHEAD: usize = NONCE_SIZE + TAG_SIZE;

/// Fixed salt for HKDF key derivation (version 1).
///
/// Using a fixed, domain-specific salt ensures that DEK derivation is
/// deterministic across versions. Changing this salt would invalidate all
/// previously derived DEKs, making encrypted data undecryptable. If the
/// derivation scheme ever needs to change, introduce a new versioned
/// constant (e.g., `HKDF_SALT_V2`) and a migration path.
const HKDF_SALT_V1: &[u8] = b"grafeo-hkdf-v1";

// -------------------------------------------------------------------------
// PageEncryptor
// -------------------------------------------------------------------------

/// Encrypts and decrypts data using AES-256-GCM.
///
/// Each call requires a nonce (12 bytes) and associated authenticated data (AAD).
/// The AAD binds the ciphertext to its storage location, preventing relocation attacks.
///
/// This type does not know what it's encrypting: storage components provide their own
/// nonce and AAD based on their natural counters (LSN, page number, chunk sequence).
pub struct PageEncryptor {
    cipher: Aes256Gcm,
}

struct WipingBuffer(Vec<u8>);

impl WipingBuffer {
    fn try_with_capacity(capacity: usize, allocation_bound: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Storage(StorageError::Full))?;
        if bytes.capacity() > allocation_bound {
            return Err(Error::Storage(StorageError::Full));
        }
        Ok(Self(bytes))
    }

    fn into_vec(mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

impl Drop for WipingBuffer {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl PageEncryptor {
    /// Creates a new encryptor from a 32-byte key.
    ///
    /// # Panics
    ///
    /// Panics if the key length is not 32 bytes (cannot happen when called
    /// with `&[u8; KEY_SIZE]`).
    #[must_use]
    pub fn new(key: &[u8; KEY_SIZE]) -> Self {
        Self {
            cipher: Aes256Gcm::new_from_slice(key).expect("AES-256-GCM key is always 32 bytes"),
        }
    }

    /// Conservative heap-capacity bound for [`Self::encrypt`].
    ///
    /// Encryption uses one fallibly allocated output buffer. Doubling the
    /// exact requested capacity follows Grafeo's allocator-slack convention
    /// and the implementation rejects an observed capacity above this bound.
    #[must_use]
    pub const fn encrypt_allocation_bound(plaintext_len: usize) -> Option<usize> {
        let Some(stored_len) = plaintext_len.checked_add(ENCRYPTION_OVERHEAD) else {
            return None;
        };
        stored_len.checked_mul(2)
    }

    /// Conservative heap-capacity bound for [`Self::decrypt`].
    ///
    /// Decryption copies only the ciphertext bytes into one fallibly allocated
    /// in-place buffer. Inputs shorter than a nonce plus tag are invalid.
    #[must_use]
    pub const fn decrypt_allocation_bound(encrypted_len: usize) -> Option<usize> {
        let Some(plaintext_len) = encrypted_len.checked_sub(ENCRYPTION_OVERHEAD) else {
            return None;
        };
        plaintext_len.checked_mul(2)
    }

    /// Encrypts plaintext with the given nonce and AAD.
    ///
    /// Returns `nonce || ciphertext || tag`.
    ///
    /// # Errors
    ///
    /// Returns an error if encryption fails (should not happen with valid inputs).
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        nonce: &[u8; NONCE_SIZE],
        aad: &[u8],
    ) -> Result<Vec<u8>> {
        let allocation_bound = Self::encrypt_allocation_bound(plaintext.len())
            .ok_or(Error::Storage(StorageError::Full))?;
        let stored_len = plaintext
            .len()
            .checked_add(ENCRYPTION_OVERHEAD)
            .ok_or(Error::Storage(StorageError::Full))?;
        let mut output = WipingBuffer::try_with_capacity(stored_len, allocation_bound)?;
        output.0.extend_from_slice(nonce);
        output.0.extend_from_slice(plaintext);

        let nonce_obj = Nonce::from_slice(nonce);
        let tag = self
            .cipher
            .encrypt_in_place_detached(nonce_obj, aad, &mut output.0[NONCE_SIZE..])
            .map_err(|_| Error::Crypto(CryptoError::EncryptionFailed))?;
        output.0.extend_from_slice(tag.as_slice());
        debug_assert_eq!(output.0.len(), stored_len);
        Ok(output.into_vec())
    }

    /// Decrypts data produced by [`encrypt`](Self::encrypt).
    ///
    /// Input format: `nonce(12) || ciphertext || tag(16)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short, the authentication tag is invalid
    /// (wrong key, tampered ciphertext, or wrong AAD), or decryption fails.
    pub fn decrypt(&self, encrypted: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < NONCE_SIZE + TAG_SIZE {
            return Err(Error::Crypto(CryptoError::CiphertextTooShort));
        }

        let allocation_bound = Self::decrypt_allocation_bound(encrypted.len())
            .ok_or(Error::Storage(StorageError::Full))?;
        let plaintext_len = encrypted.len() - ENCRYPTION_OVERHEAD;
        let (nonce_bytes, ciphertext_and_tag) = encrypted.split_at(NONCE_SIZE);
        let (ciphertext, tag_bytes) = ciphertext_and_tag.split_at(plaintext_len);
        let nonce = Nonce::from_slice(nonce_bytes);
        let tag = Tag::from_slice(tag_bytes);
        debug_assert_eq!(tag_bytes.len(), TAG_SIZE);

        let mut plaintext = WipingBuffer::try_with_capacity(plaintext_len, allocation_bound)?;
        plaintext.0.extend_from_slice(ciphertext);

        self.cipher
            .decrypt_in_place_detached(nonce, aad, &mut plaintext.0, tag)
            .map_err(|_| Error::Crypto(CryptoError::AuthenticationFailed))?;
        Ok(plaintext.into_vec())
    }

    /// Authenticates and decrypts into an exact caller-owned destination.
    ///
    /// This path performs no heap allocation. `plaintext` must be exactly the
    /// ciphertext length after removing the nonce and authentication tag. On
    /// every error, including preflight length errors, the complete destination
    /// is zeroized so stale bytes cannot be mistaken for authenticated output.
    ///
    /// # Errors
    ///
    /// Returns an allocation-free crypto error when the stored bytes are too
    /// short, the destination length is not exact, or authentication fails.
    pub fn decrypt_into(&self, encrypted: &[u8], aad: &[u8], plaintext: &mut [u8]) -> Result<()> {
        let Some(plaintext_len) = encrypted.len().checked_sub(ENCRYPTION_OVERHEAD) else {
            plaintext.zeroize();
            return Err(Error::Crypto(CryptoError::CiphertextTooShort));
        };
        if plaintext.len() != plaintext_len {
            plaintext.zeroize();
            return Err(Error::Crypto(CryptoError::PlaintextLengthMismatch));
        }

        let (nonce_bytes, ciphertext_and_tag) = encrypted.split_at(NONCE_SIZE);
        let (ciphertext, tag_bytes) = ciphertext_and_tag.split_at(plaintext_len);
        let nonce = Nonce::from_slice(nonce_bytes);
        let tag = Tag::from_slice(tag_bytes);
        debug_assert_eq!(tag_bytes.len(), TAG_SIZE);
        plaintext.copy_from_slice(ciphertext);

        if self
            .cipher
            .decrypt_in_place_detached(nonce, aad, plaintext, tag)
            .is_err()
        {
            plaintext.zeroize();
            return Err(Error::Crypto(CryptoError::AuthenticationFailed));
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------
// KeyChain
// -------------------------------------------------------------------------

/// Manages the master encryption key and derives per-component data encryption keys.
///
/// DEKs are derived deterministically via HKDF-SHA256, so they don't need separate
/// storage. `KeyChain::derive_dek("wal", &generation_bytes)` always produces the
/// same key for the same ME and inputs.
pub struct KeyChain {
    me: Zeroizing<[u8; KEY_SIZE]>,
}

impl KeyChain {
    /// Creates a key chain from a master encryption key.
    #[must_use]
    pub fn new(me: [u8; KEY_SIZE]) -> Self {
        Self {
            me: Zeroizing::new(me),
        }
    }

    /// Derives a data encryption key for the given context and component ID.
    ///
    /// The `context` identifies the storage component (e.g., `"grafeo-wal"`,
    /// `"grafeo-pages"`). The `id` is component-specific (e.g., WAL generation,
    /// file ID, snapshot ID).
    ///
    /// # Panics
    ///
    /// Panics if HKDF expansion fails for a 32-byte output (cannot happen with
    /// SHA-256, which supports up to 255 * 32 = 8160 bytes).
    #[must_use]
    pub fn derive_dek(&self, context: &str, id: &[u8]) -> Zeroizing<[u8; KEY_SIZE]> {
        let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT_V1), &*self.me);
        let mut dek = Zeroizing::new([0u8; KEY_SIZE]);
        hk.expand_multi_info(&[context.as_bytes(), id], &mut *dek)
            .expect("HKDF-SHA256 output length is valid for 32 bytes");
        dek
    }

    /// Creates a [`PageEncryptor`] for the given context and component ID.
    #[must_use]
    pub fn encryptor_for(&self, context: &str, id: &[u8]) -> PageEncryptor {
        let dek = self.derive_dek(context, id);
        PageEncryptor::new(&dek)
    }
}

// -------------------------------------------------------------------------
// KeyProvider
// -------------------------------------------------------------------------

/// Source for the root encryption key.
///
/// Built-in implementations:
/// - [`PasswordKeyProvider`]: derives key from a passphrase via Argon2id
/// - [`RawKeyProvider`]: uses a pre-existing 32-byte key directly
pub trait KeyProvider: Send + Sync {
    /// Provides the root key used to unwrap the master encryption key.
    ///
    /// # Errors
    ///
    /// Returns an error if the key cannot be obtained (missing file, bad env var, etc.).
    fn provide_root_key(&self) -> Result<Zeroizing<[u8; KEY_SIZE]>>;
}

/// Derives the root key from a passphrase using Argon2id.
///
/// Memory: 64 MiB, iterations: 3, parallelism: 1.
/// Takes ~300ms on modern hardware.
pub struct PasswordKeyProvider {
    password: Zeroizing<Vec<u8>>,
}

impl PasswordKeyProvider {
    /// Creates a new password-based key provider.
    #[must_use]
    pub fn new(password: impl Into<Vec<u8>>) -> Self {
        Self {
            password: Zeroizing::new(password.into()),
        }
    }

    /// Derives the root key from the password and salt using Argon2id.
    ///
    /// # Errors
    ///
    /// Returns an error if key derivation fails.
    pub fn derive_with_salt(&self, salt: &[u8]) -> Result<Zeroizing<[u8; KEY_SIZE]>> {
        use argon2::{Algorithm, Argon2, Params, Version};

        let params = Params::new(64 * 1024, 3, 1, Some(KEY_SIZE))
            .map_err(|e| Error::Internal(format!("argon2 params: {e}")))?;
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        let mut key = Zeroizing::new([0u8; KEY_SIZE]);
        argon2
            .hash_password_into(self.password.as_slice(), salt, &mut *key)
            .map_err(|e| Error::Internal(format!("argon2 key derivation: {e}")))?;
        Ok(key)
    }
}

impl KeyProvider for PasswordKeyProvider {
    fn provide_root_key(&self) -> Result<Zeroizing<[u8; KEY_SIZE]>> {
        Err(Error::InvalidValue(
            "password-based key derivation requires a stored salt; \
             use derive_with_salt() instead"
                .to_string(),
        ))
    }
}

/// Provides a raw 32-byte key directly (from a file, env var, or HSM).
pub struct RawKeyProvider {
    key: Zeroizing<[u8; KEY_SIZE]>,
}

impl RawKeyProvider {
    /// Creates a provider from a raw 32-byte key.
    #[must_use]
    pub fn new(key: [u8; KEY_SIZE]) -> Self {
        Self {
            key: Zeroizing::new(key),
        }
    }
}

impl KeyProvider for RawKeyProvider {
    fn provide_root_key(&self) -> Result<Zeroizing<[u8; KEY_SIZE]>> {
        Ok(self.key.clone())
    }
}

// -------------------------------------------------------------------------
// ME wrapping
// -------------------------------------------------------------------------

/// Wraps (encrypts) a master encryption key with the root key using AES-256-GCM.
///
/// Returns `nonce(12) || ciphertext(32) || tag(16)` = 60 bytes total.
///
/// # Errors
///
/// Returns an error if encryption fails.
pub fn wrap_me(root_key: &[u8; KEY_SIZE], me: &[u8; KEY_SIZE]) -> Result<Vec<u8>> {
    let encryptor = PageEncryptor::new(root_key);
    // Generate a random nonce so every wrap produces unique ciphertext,
    // even if the same root key and ME are used more than once.
    let mut nonce = [0u8; NONCE_SIZE];
    rand::rng().fill(&mut nonce);
    encryptor.encrypt(me, &nonce, b"grafeo-me-wrap")
}

/// Unwraps (decrypts) a master encryption key using the root key.
///
/// Input: the 60-byte blob produced by [`wrap_me`].
///
/// # Errors
///
/// Returns an error if the root key is wrong or the wrapped ME is corrupted.
pub fn unwrap_me(root_key: &[u8; KEY_SIZE], wrapped: &[u8]) -> Result<Zeroizing<[u8; KEY_SIZE]>> {
    let encryptor = PageEncryptor::new(root_key);
    let plaintext = encryptor.decrypt(wrapped, b"grafeo-me-wrap")?;
    if plaintext.len() != KEY_SIZE {
        return Err(Error::Internal(format!(
            "unwrapped ME has wrong length: expected {KEY_SIZE}, got {}",
            plaintext.len()
        )));
    }
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    key.copy_from_slice(&plaintext);
    Ok(key)
}

// -------------------------------------------------------------------------
// Nonce helpers
// -------------------------------------------------------------------------

/// Builds a 12-byte nonce from a 4-byte high part and an 8-byte low part.
///
/// This is the standard layout for counter-based nonces in Grafeo:
/// `high(4) || low(8)` where `high` is a generation/file ID and `low` is
/// a monotonic counter (LSN, page number, chunk sequence).
#[must_use]
pub fn build_nonce(high: u32, low: u64) -> [u8; NONCE_SIZE] {
    let mut nonce = [0u8; NONCE_SIZE];
    nonce[..4].copy_from_slice(&high.to_be_bytes());
    nonce[4..].copy_from_slice(&low.to_be_bytes());
    nonce
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

// Miri cannot interpret AES-NI / CLMUL intrinsics used by aes-gcm,
// falling back to a software path that takes hours. Skip under Miri.
#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    fn test_key() -> [u8; KEY_SIZE] {
        let mut key = [0u8; KEY_SIZE];
        for (i, byte) in key.iter_mut().enumerate() {
            // reason: KEY_SIZE is 32, index fits u8
            #[allow(clippy::cast_possible_truncation)]
            {
                *byte = i as u8;
            }
        }
        key
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"Alix knows Gus";
        let nonce = build_nonce(1, 42);
        let aad = b"wal_segment";

        let encrypted = encryptor.encrypt(plaintext, &nonce, aad).unwrap();
        assert_ne!(&encrypted[NONCE_SIZE..], plaintext);

        let decrypted = encryptor.decrypt(&encrypted, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_into_authenticates_into_the_exact_caller_buffer() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"exact caller-owned plaintext";
        let nonce = build_nonce(11, 29);
        let aad = b"qualified spill frame";
        let encrypted = encryptor.encrypt(plaintext, &nonce, aad).unwrap();
        let mut destination = vec![0xA5; plaintext.len()];

        encryptor
            .decrypt_into(&encrypted, aad, &mut destination)
            .unwrap();

        assert_eq!(destination, plaintext);
        assert_eq!(destination.len(), plaintext.len());
    }

    #[test]
    fn decrypt_into_zeroizes_destination_on_wrong_key_aad_or_ciphertext() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"authenticated exact output";
        let nonce = build_nonce(12, 30);
        let aad = b"correct aad";
        let encrypted = encryptor.encrypt(plaintext, &nonce, aad).unwrap();

        let mut wrong_key = test_key();
        wrong_key[0] ^= 0x80;
        let wrong_encryptor = PageEncryptor::new(&wrong_key);
        let mut wrong_key_output = vec![0xA5; plaintext.len()];
        assert!(
            wrong_encryptor
                .decrypt_into(&encrypted, aad, &mut wrong_key_output)
                .is_err()
        );
        assert_eq!(wrong_key_output, vec![0; plaintext.len()]);

        let mut wrong_aad_output = vec![0xA5; plaintext.len()];
        assert!(
            encryptor
                .decrypt_into(&encrypted, b"wrong aad", &mut wrong_aad_output)
                .is_err()
        );
        assert_eq!(wrong_aad_output, vec![0; plaintext.len()]);

        let mut tampered = encrypted;
        tampered[NONCE_SIZE] ^= 0x40;
        let mut tampered_output = vec![0xA5; plaintext.len()];
        assert!(
            encryptor
                .decrypt_into(&tampered, aad, &mut tampered_output)
                .is_err()
        );
        assert_eq!(tampered_output, vec![0; plaintext.len()]);
    }

    #[test]
    fn decrypt_into_zeroizes_destination_on_every_preflight_error() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"length-bound plaintext";
        let nonce = build_nonce(13, 31);
        let encrypted = encryptor.encrypt(plaintext, &nonce, b"aad").unwrap();

        let mut short_destination = vec![0xA5; plaintext.len() - 1];
        let error = encryptor
            .decrypt_into(&encrypted, b"aad", &mut short_destination)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Crypto(CryptoError::PlaintextLengthMismatch)
        ));
        assert_eq!(short_destination, vec![0; plaintext.len() - 1]);

        let mut long_destination = vec![0xA5; plaintext.len() + 1];
        let error = encryptor
            .decrypt_into(&encrypted, b"aad", &mut long_destination)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Crypto(CryptoError::PlaintextLengthMismatch)
        ));
        assert_eq!(long_destination, vec![0; plaintext.len() + 1]);

        let truncated = vec![0xCC; ENCRYPTION_OVERHEAD - 1];
        let mut truncated_output = vec![0xA5; 7];
        assert!(
            encryptor
                .decrypt_into(&truncated, b"aad", &mut truncated_output)
                .is_err()
        );
        assert_eq!(truncated_output, vec![0; 7]);
    }

    #[test]
    fn crypto_buffers_respect_reported_allocation_bounds() {
        let encryptor = PageEncryptor::new(&test_key());
        for plaintext_len in [0, 1, 31, 1024] {
            let plaintext = vec![0xA5; plaintext_len];
            let nonce = build_nonce(7, plaintext_len as u64);
            let encrypted = encryptor.encrypt(&plaintext, &nonce, b"bounded").unwrap();
            assert!(
                encrypted.capacity()
                    <= PageEncryptor::encrypt_allocation_bound(plaintext_len).unwrap()
            );

            let decrypted = encryptor.decrypt(&encrypted, b"bounded").unwrap();
            assert!(
                decrypted.capacity()
                    <= PageEncryptor::decrypt_allocation_bound(encrypted.len()).unwrap()
            );
            assert_eq!(decrypted, plaintext);
        }
    }

    #[test]
    fn crypto_allocation_bounds_reject_invalid_lengths() {
        assert_eq!(PageEncryptor::encrypt_allocation_bound(usize::MAX), None);
        assert_eq!(PageEncryptor::decrypt_allocation_bound(0), None);
        assert_eq!(
            PageEncryptor::decrypt_allocation_bound(ENCRYPTION_OVERHEAD - 1),
            None
        );
    }

    #[test]
    fn wrong_key_fails() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"secret data";
        let nonce = build_nonce(0, 0);

        let encrypted = encryptor.encrypt(plaintext, &nonce, b"aad").unwrap();

        let mut wrong_key = test_key();
        wrong_key[0] ^= 0xFF;
        let wrong_encryptor = PageEncryptor::new(&wrong_key);
        assert!(wrong_encryptor.decrypt(&encrypted, b"aad").is_err());
    }

    #[test]
    fn wrong_aad_fails() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"secret data";
        let nonce = build_nonce(0, 0);

        let encrypted = encryptor
            .encrypt(plaintext, &nonce, b"correct_aad")
            .unwrap();
        assert!(encryptor.decrypt(&encrypted, b"wrong_aad").is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = b"secret data";
        let nonce = build_nonce(0, 0);

        let mut encrypted = encryptor.encrypt(plaintext, &nonce, b"aad").unwrap();
        // Flip a byte in the ciphertext
        let mid = encrypted.len() / 2;
        encrypted[mid] ^= 0xFF;
        assert!(encryptor.decrypt(&encrypted, b"aad").is_err());
    }

    #[test]
    fn truncated_data_fails() {
        let encryptor = PageEncryptor::new(&test_key());
        // Too short: less than nonce + tag
        let short = vec![0u8; NONCE_SIZE + TAG_SIZE - 1];
        assert!(encryptor.decrypt(&short, b"aad").is_err());
    }

    #[test]
    fn key_derivation_deterministic() {
        let chain = KeyChain::new(test_key());
        let dek1 = chain.derive_dek("grafeo-wal", &42u64.to_be_bytes());
        let dek2 = chain.derive_dek("grafeo-wal", &42u64.to_be_bytes());
        assert_eq!(*dek1, *dek2, "same inputs must produce same DEK");
    }

    #[test]
    fn allocation_free_key_derivation_preserves_concatenated_info() {
        let key = test_key();
        let chain = KeyChain::new(key);
        let context = b"grafeo-spill-record-v1";
        let id = [0x5A; 32];
        let derived = chain.derive_dek(std::str::from_utf8(context).unwrap(), &id);

        let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT_V1), &key);
        let mut concatenated = [0u8; 64];
        let info_len = context.len() + id.len();
        concatenated[..context.len()].copy_from_slice(context);
        concatenated[context.len()..info_len].copy_from_slice(&id);
        let mut expected = [0u8; KEY_SIZE];
        hk.expand(&concatenated[..info_len], &mut expected).unwrap();

        assert_eq!(*derived, expected);
    }

    #[test]
    fn different_contexts_produce_different_keys() {
        let chain = KeyChain::new(test_key());
        let wal_dek = chain.derive_dek("grafeo-wal", &1u64.to_be_bytes());
        let page_dek = chain.derive_dek("grafeo-pages", &1u64.to_be_bytes());
        assert_ne!(*wal_dek, *page_dek);
    }

    #[test]
    fn different_ids_produce_different_keys() {
        let chain = KeyChain::new(test_key());
        let dek1 = chain.derive_dek("grafeo-wal", &1u64.to_be_bytes());
        let dek2 = chain.derive_dek("grafeo-wal", &2u64.to_be_bytes());
        assert_ne!(*dek1, *dek2);
    }

    #[test]
    fn encryptor_for_works() {
        let chain = KeyChain::new(test_key());
        let encryptor = chain.encryptor_for("grafeo-wal", &1u64.to_be_bytes());

        let plaintext = b"WAL record payload";
        let nonce = build_nonce(1, 100);
        let encrypted = encryptor.encrypt(plaintext, &nonce, b"wal").unwrap();
        let decrypted = encryptor.decrypt(&encrypted, b"wal").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn me_wrap_unwrap_roundtrip() {
        let root_key = test_key();
        let mut me = [0u8; KEY_SIZE];
        for (i, byte) in me.iter_mut().enumerate() {
            // reason: KEY_SIZE is 32, (255 - i) fits u8
            #[allow(clippy::cast_possible_truncation)]
            {
                *byte = (255 - i) as u8;
            }
        }

        let wrapped = wrap_me(&root_key, &me).unwrap();
        assert_eq!(wrapped.len(), NONCE_SIZE + KEY_SIZE + TAG_SIZE);

        let unwrapped = unwrap_me(&root_key, &wrapped).unwrap();
        assert_eq!(*unwrapped, me);
    }

    #[test]
    fn me_wrap_uses_random_nonce() {
        let root_key = test_key();
        let me = [42u8; KEY_SIZE];

        let wrapped1 = wrap_me(&root_key, &me).unwrap();
        let wrapped2 = wrap_me(&root_key, &me).unwrap();

        // Random nonces mean the wrapped output differs each time
        assert_ne!(
            wrapped1, wrapped2,
            "wrap_me must produce different ciphertext on each call"
        );

        // Both must still unwrap to the same ME
        let unwrapped1 = unwrap_me(&root_key, &wrapped1).unwrap();
        let unwrapped2 = unwrap_me(&root_key, &wrapped2).unwrap();
        assert_eq!(*unwrapped1, me);
        assert_eq!(*unwrapped2, me);
    }

    #[test]
    fn me_unwrap_wrong_key_fails() {
        let root_key = test_key();
        let me = [42u8; KEY_SIZE];
        let wrapped = wrap_me(&root_key, &me).unwrap();

        let mut wrong_root = root_key;
        wrong_root[0] ^= 0xFF;
        assert!(unwrap_me(&wrong_root, &wrapped).is_err());
    }

    #[test]
    fn build_nonce_layout() {
        let nonce = build_nonce(0x0102_0304, 0x0506_0708_090A_0B0C);
        assert_eq!(nonce[0..4], [0x01, 0x02, 0x03, 0x04]);
        assert_eq!(
            nonce[4..12],
            [0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]
        );
    }

    #[test]
    fn password_key_derivation() {
        let provider = PasswordKeyProvider::new(b"test-password-123");
        let salt = [1u8; 16];
        let key1 = provider.derive_with_salt(&salt).unwrap();
        let key2 = provider.derive_with_salt(&salt).unwrap();
        assert_eq!(*key1, *key2, "same password + salt must produce same key");

        let different_salt = [2u8; 16];
        let key3 = provider.derive_with_salt(&different_salt).unwrap();
        assert_ne!(*key1, *key3, "different salts must produce different keys");
    }

    #[test]
    fn password_provider_provide_root_key_returns_error() {
        let provider = PasswordKeyProvider::new(b"test-password");
        let result = provider.provide_root_key();
        assert!(
            result.is_err(),
            "provide_root_key must fail for password providers"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("salt"),
            "error should mention salt requirement, got: {err_msg}"
        );
    }

    #[test]
    fn raw_key_provider() {
        let key = test_key();
        let provider = RawKeyProvider::new(key);
        let provided = provider.provide_root_key().unwrap();
        assert_eq!(*provided, key);
    }

    #[test]
    fn empty_plaintext_roundtrip() {
        let encryptor = PageEncryptor::new(&test_key());
        let nonce = build_nonce(0, 0);
        let encrypted = encryptor.encrypt(b"", &nonce, b"").unwrap();
        let decrypted = encryptor.decrypt(&encrypted, b"").unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn large_payload_roundtrip() {
        let encryptor = PageEncryptor::new(&test_key());
        let plaintext = vec![0xABu8; 1024 * 1024]; // 1 MiB
        let nonce = build_nonce(0, 0);
        let encrypted = encryptor.encrypt(&plaintext, &nonce, b"snapshot").unwrap();
        let decrypted = encryptor.decrypt(&encrypted, b"snapshot").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn hkdf_uses_fixed_salt_not_none() {
        // Verify that derive_dek uses a fixed salt, not None.
        // Derive a DEK with the real code (which uses HKDF_SALT_V1),
        // then derive one manually with None salt: they must differ.
        let me = test_key();
        let chain = KeyChain::new(me);
        let dek_with_salt = chain.derive_dek("grafeo-wal", &1u64.to_be_bytes());

        // Manual derivation with None salt (the old, broken behavior)
        let hk_no_salt = Hkdf::<Sha256>::new(None, &me);
        let mut info = Vec::new();
        info.extend_from_slice(b"grafeo-wal");
        info.extend_from_slice(&1u64.to_be_bytes());
        let mut dek_no_salt = [0u8; KEY_SIZE];
        hk_no_salt.expand(&info, &mut dek_no_salt).unwrap();

        assert_ne!(
            *dek_with_salt, dek_no_salt,
            "derive_dek must use a fixed salt, not None"
        );
    }
}
