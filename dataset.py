import os
import subprocess
import hashlib
import random
import string
import csv
import time
import sys
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

# --- 1. Setup and Dependencies ---
DATABASE_FILE = "training_data.csv"
GDB_OUTPUT_FILE_PREFIX = "memory_dump_" 
NUM_CYCLES = 1000 
FILE_TYPES = ['.txt', '.docx', '.xlsx', '.pdf', '.jpg']

# Debug purposes: check sudo permissions (redundant now)
if os.geteuid() == 0:
    print("Script is being run with sudo permissions.")
else:
    print("Script is being run without sudo permissions.")

def initialize_database(file_path):
    """Initializes the database by writing the header if the file does not exists."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["memory_dump_file", "key", "key_address", "is_synthetic", "ransomware_behavior", "file_type", "file_size", "key_offset"])  # Add more fields as needed
    else:
        print(f"Database file {file_path} exists, skipping initialization.")

def generate_random_file(size, output_path):
    """Generates a random file of the specified size and type."""
    file_type = os.path.splitext(output_path)[1]
    if file_type == '.txt':
        with open(output_path, 'w') as file:
            random_text = ''.join(random.choice(string.ascii_letters + string.digits + ' ') for _ in range(size))
            file.write(random_text)
    else:
        with open(output_path, 'wb') as file:
            file.write(os.urandom(size))
    return output_path

def generate_random_key(size_in_bytes):
    """Generate random key"""
    return os.urandom(size_in_bytes)

def encrypt_file(file_path, key, behavior="partial"):
    """Encrypts a file using AES, with different behaviours."""
    with open(file_path, 'rb') as file:
        plaintext = file.read()

    encryption_algorithm = random.choice(["AES256", "AES128"])

    if encryption_algorithm == "AES256":
        cipher = Cipher(algorithms.AES(key), modes.CBC(os.urandom(16)), backend=default_backend())
    elif encryption_algorithm == "AES128":
        cipher = Cipher(algorithms.AES(key), modes.CBC(os.urandom(16)), backend=default_backend())
    else:
        print(f"Invalid encryption algorithm: {encryption_algorithm}")
        sys.exit(1)
        
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    encryptor = cipher.encryptor()
    
    if behavior == "partial":
        # Simulate partial encryption
        partial_size = random.randint(0, len(plaintext))
        padded_plaintext = padder.update(plaintext[:partial_size]) + padder.finalize()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        # Combine the encrypted and plain text to make it partial
        ciphertext_with_plaintext = ciphertext + plaintext[partial_size:]
    elif behavior == "full":
        padded_plaintext = padder.update(plaintext) + padder.finalize()
        ciphertext_with_plaintext = encryptor.update(padded_plaintext) + encryptor.finalize()
    elif behavior == "key_wrapping":
        # Encrypt the key
        key_wrapper = generate_random_key(32)
        key_cipher = Cipher(algorithms.AES(key_wrapper), modes.CBC(os.urandom(16)), backend=default_backend())
        key_padder = padding.PKCS7(algorithms.AES.block_size).padder()
        key_padded = key_padder.update(key) + key_padder.finalize()
        key_encryptor = key_cipher.encryptor()
        encrypted_key = key_encryptor.update(key_padded) + key_encryptor.finalize()
        # encrypt the file
        padded_plaintext = padder.update(plaintext) + padder.finalize()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        # combine the ciphertext and the encrypted key
        ciphertext_with_plaintext = encrypted_key + ciphertext

    encrypted_path = file_path + ".enc"
    with open(encrypted_path, 'wb') as file:
        file.write(ciphertext_with_plaintext)
        
    return encrypted_path, encryption_algorithm

def get_memory_dump_gdb(pid, output_file):
    """Creates a memory dump of the process using gdb."""
    # Construct the gdb command
    gdb_command = [
        "gdb",
        "-batch",
        "-ex", f"attach {pid}",
        "-ex", "gcore " + output_file,
        "-ex", "detach",
        "-ex", "quit"
    ]
    # Run gdb
    process = subprocess.Popen(gdb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error getting memory dump using gdb:")
        print(stderr.decode())
        print("Please check if gdb is installed, and you have correct permissions to access the process.")
        return False
    else:
        print("Memory dump created successfully.")
        return True

def generate_synthetic_memory_dump(size, key, offset):
    """Generates a synthetic memory dump with a key at a given offset."""
    dump = bytearray(os.urandom(size))
    if offset + len(key) <= len(dump):
        dump[offset:offset+len(key)] = key
    else:
        print(f"Warning key out of bounds of synthetic memory dump size of {len(dump)}")
    return bytes(dump)

def find_key_in_dump(dump_file, key):
    """Finds all occurances of the key within the dump file."""
    key_found = False
    addresses = []
    with open(dump_file, "rb") as f:
        mem_dump = f.read()
    start = 0
    while True:
        start = mem_dump.find(key, start)
        if start == -1:
           break
        else:
          key_found = True
          addresses.append(start)
          start +=1
    if key_found:
      print(f"Key found at location(s): {addresses}")
    else:
       print(f"Key not found in memory dump: {dump_file}")
    return addresses

def write_to_database(database_file, memory_dump_file, key, key_address, is_synthetic, ransomware_behavior, file_type, file_size, key_offset):
    """Writes the data to the database (CSV for simplicity)."""
    with open(database_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([memory_dump_file, key.hex(), str(key_address), is_synthetic, ransomware_behavior, file_type, file_size, key_offset])

def get_process_id():
    """Gets the process ID of the current script."""
    return os.getpid()

def unload_key(key):
    """Unloads the key from memory, by setting to None."""
    key = None
    return key

def create_memory_variation():
    """Creates memory variation by allocating/deallocating random objects."""
    num_objects = random.randint(5, 15)
    objects = []

    for _ in range(num_objects):
        type = random.choice(["str", "list", "dict"])
        if type == "str":
            objects.append("a"* random.randint(10, 100))
        elif type == "list":
           objects.append(list(range(random.randint(5, 50))))
        elif type == "dict":
           objects.append({str(i): i for i in range(random.randint(1, 30))})
    
    if random.random() > 0.5:
        for _ in range(random.randint(0, len(objects))):
            if objects:
              objects.pop()
    time.sleep(random.uniform(0, 0.2))
    return objects

def generate_file_info():
    """Generate random file size and type."""
    file_type = random.choice(FILE_TYPES)
    file_size = random.randint(50*1024, 10*1024*1024)  # between 50kb and 10mb
    return file_type, file_size
    
# --- 2. File Creation and Encryption ---
# Initialize database
initialize_database(DATABASE_FILE)

# --- 6. Repetition and Control ---
dataset_stats = {
  'synthetic_count': 0,
  'real_count': 0,
  'behavior_counts': {},
  'file_type_counts': {},
  'total_dataset_size_bytes': 0
}
memory_dump_count = 0

for cycle in range(NUM_CYCLES):
    print(f"Starting cycle {cycle+1} of {NUM_CYCLES}")
    
    # Generate key
    key_size = random.choice([16, 32]) # 16 for AES128, 32 for AES256
    key = generate_random_key(key_size)
    
    # --- Memory Variation ---
    objects_in_mem = create_memory_variation()
    
    # Generate test file info
    file_type, file_size = generate_file_info()
    test_file_path = f"test_file_{cycle}{file_type}"
    
    # Generate test file
    test_file_path = generate_random_file(file_size, test_file_path)

    # Simulate ransomware behaviour
    ransomware_behavior = random.choice(["partial", "full", "key_wrapping"])

    # Encrypt the test file
    encrypted_file_path, _ = encrypt_file(test_file_path, key, ransomware_behavior)
    
    memory_dump_file = f"{GDB_OUTPUT_FILE_PREFIX}{memory_dump_count}.bin"

    if cycle % 2 == 0:  # Even cycles generate synthetic data
      print("Generating synthetic data...")
      is_synthetic = True
      # --- 3a. Generate Synthetic Memory Dump ---
      synthetic_dump_size = random.randint(2 * 1024 * 1024, 10 * 1024 * 1024)
      key_offset = random.randint(0, synthetic_dump_size - len(key))
      synthetic_dump = generate_synthetic_memory_dump(synthetic_dump_size, key, key_offset)
      with open(memory_dump_file, "wb") as f:
         f.write(synthetic_dump)

      # --- 4. Key Location Finding ---
      key_addresses = find_key_in_dump(memory_dump_file, key)

      # --- 5. Data Export ---
      write_to_database(DATABASE_FILE, memory_dump_file, key, key_addresses, is_synthetic, ransomware_behavior, file_type, file_size, key_offset)
      
      dataset_stats['synthetic_count'] += 1
      dataset_stats['total_dataset_size_bytes'] += synthetic_dump_size
      if ransomware_behavior not in dataset_stats['behavior_counts']:
         dataset_stats['behavior_counts'][ransomware_behavior] = 0
      dataset_stats['behavior_counts'][ransomware_behavior] += 1
      if file_type not in dataset_stats['file_type_counts']:
        dataset_stats['file_type_counts'][file_type] = 0
      dataset_stats['file_type_counts'][file_type] +=1
      memory_dump_count += 1

      # Remove key from memory
      key = unload_key(key)
    else: # Odd cycles generate real dumps
        print("Generating real memory dump...")
        is_synthetic = False
        # --- 3. Memory Dumping ---
        # Get the process id of the script
        process_id = get_process_id()
        # Dump the memory to a file
        if get_memory_dump_gdb(process_id, memory_dump_file):
        # --- 4. Key Location Finding ---
          key_addresses = find_key_in_dump(memory_dump_file, key)
          # --- 5. Data Export ---
          write_to_database(DATABASE_FILE, memory_dump_file, key, key_addresses, is_synthetic, ransomware_behavior, file_type, file_size, 0)
          dataset_stats['real_count'] += 1
          dataset_stats['total_dataset_size_bytes'] += os.path.getsize(memory_dump_file)
          if ransomware_behavior not in dataset_stats['behavior_counts']:
            dataset_stats['behavior_counts'][ransomware_behavior] = 0
          dataset_stats['behavior_counts'][ransomware_behavior] += 1
          if file_type not in dataset_stats['file_type_counts']:
            dataset_stats['file_type_counts'][file_type] = 0
          dataset_stats['file_type_counts'][file_type] +=1
          memory_dump_count += 1

          # Remove key from memory
          key = unload_key(key)
        else:
          print("Skipping exporting data...")

    # --- Housekeeping ---
    # Remove test files
    os.remove(test_file_path)
    os.remove(encrypted_file_path)
    print(f"Completed cycle {cycle+1} of {NUM_CYCLES}")

# --- 7. Statistics Output ---
print("\n--- Dataset Generation Statistics ---")
print(f"Total Dataset Size: {dataset_stats['total_dataset_size_bytes']/(1024*1024):.2f} MB")
print(f"Number of Synthetic Dumps: {dataset_stats['synthetic_count']}")
print(f"Number of Real Dumps: {dataset_stats['real_count']}")
print(f"Ransomware Behavior Distribution: {dataset_stats['behavior_counts']}")
print(f"File Type Distribution: {dataset_stats['file_type_counts']}")
print(f"Successfully completed generating training data")
