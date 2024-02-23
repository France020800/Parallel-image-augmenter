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

if __name__ == '__main__':
    times = []
    for i in range(1, 10):
        time = run_parallel(i)
        times.append(time)
        print(f'Time for {i} threads: {time}')

    plt.plot(range(1, 10), times, marker='o')
    plt.xlabel('Number of threads')
    plt.ylabel('Time')
    plt.show()


