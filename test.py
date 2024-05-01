import threading
import time

# Function to generate a number every second
def generate_number(numbers, lock, exit_flag):
    count = 0
    while not exit_flag.is_set():
        with lock:
            numbers.clear()  # Clear the list before appending the new number
            numbers.append(count)
            print("Generated number:", count)
            count += 1
        time.sleep(1)

# Function to print a number every 2 seconds
def print_number(numbers, lock, exit_flag):
    while not exit_flag.is_set():
        time.sleep(2)  # Wait for 2 seconds
        with lock:
            if numbers:
                number = numbers[0]  # Take the first number in the list
                print("Printing number:", number)

if __name__ == "__main__":
    numbers = []
    lock = threading.Lock()
    exit_flag = threading.Event()

    generate_thread = threading.Thread(target=generate_number, args=(numbers, lock, exit_flag))
    print_thread = threading.Thread(target=print_number, args=(numbers, lock, exit_flag))

    generate_thread.start()
    print_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        exit_flag.set()
        generate_thread.join() 
        print_thread.join() 
