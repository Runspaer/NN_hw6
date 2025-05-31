#!/usr/bin/env python3
import re
import argparse
import matplotlib.pyplot as plt

LOG_LINE_RE = re.compile(r"""
    ^\s*
    (?P<epoch>\d+)\s+
    max\s*=\s*(?P<max>[-+]?\d*\.?\d+),\s*
    mean\s*=\s*(?P<mean>[-+]?\d*\.?\d+)
    \s*$
""", re.VERBOSE)

def parse_log(path):
    epochs, max_vals, mean_vals = [], [], []
    with open(path, 'r') as f:
        for line in f:
            m = LOG_LINE_RE.match(line)
            if not m:
                continue
            epochs.append(int(m.group('epoch')))
            max_vals.append(float(m.group('max')))
            mean_vals.append(float(m.group('mean')))
    return epochs, max_vals, mean_vals

def plot_schedule(epochs, max_vals, mean_vals, out_png):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, max_vals, marker='o', markersize=3, label='Максимальная награда за эпоху')
    plt.plot(epochs, mean_vals, marker='o', markersize=3, label='Средняя награда за эпоху')
    plt.xlabel('Эпоха')
    plt.ylabel('Награда')
    plt.title('График обучения')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

if __name__ == '__main__':
    '''
    Файл, который визуализирует логи, полученные из консоли обучения
    '''
    #Путь к сохранённым логам
    logfile_path = 'data/100_log_end.txt'
    #Путь к итоговому изображению
    out_png = logfile_path[:-4]+ '.png'

    epochs, max_vals, mean_vals = parse_log(logfile_path)
    plot_schedule(epochs, max_vals, mean_vals, out_png)

