#!/usr/bin/env python3
"""
Experiment 1.0: Authenticated Classical Channel

This experiment implements and tests the authenticated classical channel
required for secure QKD. Without authentication, the protocol is vulnerable
to man-in-the-middle attacks.

Key components:
1. HMAC-SHA256 based message authentication
2. Key derivation for authentication keys
3. Attack detection tests

Expected results:
- Valid messages pass authentication
- Tampered messages are rejected
- Replay attacks are detected

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import hmac
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE1_RESULTS


@dataclass
class AuthenticatedMessage:
    """Container for authenticated message."""
    content: bytes
    mac: bytes
    sender: str
    sequence_number: int = 0

    def to_dict(self):
        return {
            'content': self.content.hex(),
            'mac': self.mac.hex(),
            'sender': self.sender,
            'sequence_number': self.sequence_number
        }


class AuthenticatedChannel:
    """
    Authenticated classical channel using HMAC-SHA256.

    This provides message authentication to prevent tampering
    and man-in-the-middle attacks on the classical channel.

    For QKD:
    - First round: Uses pre-shared key
    - Subsequent rounds: Uses portion of previous QKD key
    """

    def __init__(self, shared_key: bytes, party_name: str):
        """
        Initialize authenticated channel.

        Args:
            shared_key: Shared secret key (minimum 256 bits)
            party_name: Name of this party (for identification)
        """
        if len(shared_key) < 32:
            raise ValueError("Shared key must be at least 256 bits (32 bytes)")

        self.shared_key = shared_key
        self.party_name = party_name
        self.send_counter = 0
        self.receive_counter = 0
        self.seen_sequences = set()

    def send(self, message: bytes) -> AuthenticatedMessage:
        """
        Create authenticated message.

        Args:
            message: Message content to authenticate

        Returns:
            AuthenticatedMessage with MAC
        """
        self.send_counter += 1

        # Include sequence number in MAC to prevent replay
        full_message = self.send_counter.to_bytes(8, 'big') + message
        mac = hmac.new(self.shared_key, full_message, hashlib.sha256).digest()

        return AuthenticatedMessage(
            content=message,
            mac=mac,
            sender=self.party_name,
            sequence_number=self.send_counter
        )

    def receive(self, auth_msg: AuthenticatedMessage) -> Tuple[bool, bytes, str]:
        """
        Verify and extract message.

        Args:
            auth_msg: Authenticated message to verify

        Returns:
            Tuple of (is_valid, message, error_reason)
        """
        # Check for replay attack
        if auth_msg.sequence_number in self.seen_sequences:
            return False, b'', "Replay attack detected"

        if auth_msg.sequence_number <= self.receive_counter:
            return False, b'', "Out-of-order message (possible replay)"

        # Verify MAC
        full_message = auth_msg.sequence_number.to_bytes(8, 'big') + auth_msg.content
        expected_mac = hmac.new(self.shared_key, full_message, hashlib.sha256).digest()

        if not hmac.compare_digest(auth_msg.mac, expected_mac):
            return False, b'', "MAC verification failed"

        # Update state
        self.seen_sequences.add(auth_msg.sequence_number)
        self.receive_counter = auth_msg.sequence_number

        return True, auth_msg.content, ""

    @staticmethod
    def derive_auth_key(qkd_key: bytes, purpose: str = "auth",
                        key_length: int = 32) -> bytes:
        """
        Derive authentication key from QKD key using HKDF-like construction.

        Args:
            qkd_key: Raw QKD key material
            purpose: Purpose string for domain separation
            key_length: Desired output length in bytes

        Returns:
            Derived key for authentication
        """
        # Simple HKDF-like construction
        # In production, use proper HKDF from cryptography library
        prk = hmac.new(b"QKD-AUTH", qkd_key, hashlib.sha256).digest()
        okm = hmac.new(prk, purpose.encode() + b'\x01', hashlib.sha256).digest()
        return okm[:key_length]


def test_basic_authentication():
    """Test basic message authentication."""
    print("\n" + "="*60)
    print("Test 1: Basic Authentication")
    print("="*60)

    # Create shared key
    shared_key = b'0123456789abcdef' * 2  # 32 bytes

    # Create channels for Alice and Bob
    alice_channel = AuthenticatedChannel(shared_key, "Alice")
    bob_channel = AuthenticatedChannel(shared_key, "Bob")

    # Alice sends message
    message = b"Hello Bob, here are my basis choices: ZXZXZX"
    auth_msg = alice_channel.send(message)

    print(f"Message: {message.decode()}")
    print(f"MAC: {auth_msg.mac.hex()[:32]}...")
    print(f"Sequence: {auth_msg.sequence_number}")

    # Bob receives
    valid, received, error = bob_channel.receive(auth_msg)

    print(f"\nVerification: {'PASSED' if valid else 'FAILED'}")
    if valid:
        print(f"Received: {received.decode()}")

    return valid


def test_tampered_message():
    """Test detection of tampered messages."""
    print("\n" + "="*60)
    print("Test 2: Tampered Message Detection")
    print("="*60)

    shared_key = b'0123456789abcdef' * 2

    alice_channel = AuthenticatedChannel(shared_key, "Alice")
    bob_channel = AuthenticatedChannel(shared_key, "Bob")

    # Alice sends
    message = b"My bases are: ZZZZZZ"
    auth_msg = alice_channel.send(message)
    print(f"Original message: {message.decode()}")

    # Eve tampers with the message
    tampered_content = b"My bases are: XXXXXX"  # Changed!
    tampered_msg = AuthenticatedMessage(
        content=tampered_content,
        mac=auth_msg.mac,  # Same MAC
        sender=auth_msg.sender,
        sequence_number=auth_msg.sequence_number
    )
    print(f"Tampered message: {tampered_content.decode()}")

    # Bob receives tampered message
    valid, received, error = bob_channel.receive(tampered_msg)

    print(f"\nDetection: {'FAILED - Attack NOT detected' if valid else 'SUCCESS - Attack detected'}")
    if not valid:
        print(f"Error: {error}")

    return not valid  # Success if tampering detected


def test_replay_attack():
    """Test detection of replay attacks."""
    print("\n" + "="*60)
    print("Test 3: Replay Attack Detection")
    print("="*60)

    shared_key = b'0123456789abcdef' * 2

    alice_channel = AuthenticatedChannel(shared_key, "Alice")
    bob_channel = AuthenticatedChannel(shared_key, "Bob")

    # Alice sends first message
    msg1 = alice_channel.send(b"First message")
    valid1, _, _ = bob_channel.receive(msg1)
    print(f"First message: {'Accepted' if valid1 else 'Rejected'}")

    # Eve replays the same message
    print("\nEve replays the first message...")
    valid2, _, error = bob_channel.receive(msg1)

    print(f"Replay detection: {'FAILED' if valid2 else 'SUCCESS'}")
    if not valid2:
        print(f"Error: {error}")

    return not valid2  # Success if replay detected


def test_key_derivation():
    """Test key derivation from QKD key."""
    print("\n" + "="*60)
    print("Test 4: Key Derivation")
    print("="*60)

    # Simulated QKD key
    qkd_key = bytes([np.random.randint(0, 256) for _ in range(64)])
    print(f"QKD key (first 16 bytes): {qkd_key[:16].hex()}")

    # Derive keys for different purposes
    auth_key = AuthenticatedChannel.derive_auth_key(qkd_key, "auth")
    enc_key = AuthenticatedChannel.derive_auth_key(qkd_key, "encrypt")

    print(f"Auth key: {auth_key.hex()}")
    print(f"Enc key: {enc_key.hex()}")

    # Verify keys are different
    keys_different = auth_key != enc_key
    print(f"\nKeys are different: {keys_different}")

    # Verify deterministic
    auth_key2 = AuthenticatedChannel.derive_auth_key(qkd_key, "auth")
    deterministic = auth_key == auth_key2
    print(f"Derivation is deterministic: {deterministic}")

    return keys_different and deterministic


def test_wrong_key():
    """Test authentication failure with wrong key."""
    print("\n" + "="*60)
    print("Test 5: Wrong Key Detection")
    print("="*60)

    alice_key = b'0123456789abcdef' * 2
    eve_key = b'fedcba9876543210' * 2  # Different key

    alice_channel = AuthenticatedChannel(alice_key, "Alice")
    eve_channel = AuthenticatedChannel(eve_key, "Eve")  # Wrong key!

    # Alice sends
    message = b"Secret bases"
    auth_msg = alice_channel.send(message)
    print(f"Alice sends: {message.decode()}")

    # Eve tries to verify with wrong key
    valid, _, error = eve_channel.receive(auth_msg)

    print(f"\nEve's verification: {'PASSED (BAD!)' if valid else 'FAILED (Good!)'}")
    if not valid:
        print(f"Error: {error}")

    return not valid


def run_all_tests():
    """Run all authentication tests."""
    print("="*60)
    print("EXPERIMENT 1.0: AUTHENTICATED CHANNEL")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'exp_1_0_authenticated_channel',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Run tests
    tests = [
        ('basic_authentication', test_basic_authentication),
        ('tampered_message', test_tampered_message),
        ('replay_attack', test_replay_attack),
        ('key_derivation', test_key_derivation),
        ('wrong_key', test_wrong_key),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            passed = test_func()
            results['tests'][name] = {
                'passed': passed,
                'error': None
            }
            all_passed = all_passed and passed
        except Exception as e:
            results['tests'][name] = {
                'passed': False,
                'error': str(e)
            }
            all_passed = False
            print(f"ERROR in {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    n_passed = sum(1 for t in results['tests'].values() if t['passed'])
    n_total = len(results['tests'])

    print(f"Tests passed: {n_passed}/{n_total}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    results['summary'] = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'all_passed': all_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_0_authenticated_channel.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_all_tests()
