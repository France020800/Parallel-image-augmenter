import matplotlib.pyplot as plt
import subprocess

def run_parallel(num_thread):
    result = subprocess.run(['python', 'parallel_image_augmenter.py', str(num_thread)], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    try:
        float_value = float(output)
        return float_value
    except ValueError:
        print("Error: Subprocess did not output a valid float value.")
        return None

def run_sequential():
    result = subprocess.run(['python', 'image_augmenter.py'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    try:
        float_value = float(output)
        return float_value
    except ValueError:
        print("Error: Subprocess did not output a valid float value.")
        return None

if __name__ == '__main__':
    print('Start tester')
    times = []
    time = run_sequential()
    times.append(time)
    print(f'Sequential time: {time}')
    for i in range(2, 10):
        time = run_parallel(i)
        times.append(time)
        print(f'Time for {i} threads: {time}')

    plt.plot(range(1, 10), times, marker='o')
    plt.xlabel('Number of threads')
    plt.ylabel('Time')
    plt.show()

    sequential_time = times[0]
    speedups = [sequential_time/parallel_time for parallel_time in times]

    plt.plot(range(1, 10), speedups, marker='o')
    plt.xlabel('Number of threads')
    plt.ylabel('Time')
    plt.show()
