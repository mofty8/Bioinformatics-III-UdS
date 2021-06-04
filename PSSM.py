import seaborn
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
from matplotlib import transforms
import matplotlib.patheffects
import numpy as np
f_mat = list()

class PSSM:
    def get_frequency_matrix(seq):
        f_mat = [[0 for i in range (len(seq[0]))] for j in range (4)]
        for i  in range (len(seq)):
            for j in range(len(seq[0])):
                if seq[i][j] == 'A':
                    f_mat[0][j] += 1
                elif seq[i][j] == 'C':
                    f_mat[1][j] += 1
                elif seq[i][j] == 'G':
                    f_mat[2][j] += 1
                elif seq[i][j] == 'T':
                    f_mat[3][j] += 1


        total_lenght = len(seq) * len(seq[0])
        for i  in range (len(f_mat)):
            for j in range(len(f_mat[0])):
                f_mat[i][j] /= len(seq)
        print("Frequency Matrix")
        for i in range (len(f_mat)):
            print(f_mat[i])

        return f_mat


    def get_corr_frequency_matrix(seq, prob, k):
        f_mat = [[0 for i in range (len(seq[0]))] for j in range (4)]
        for i in range(len(seq)):
            for j in range(len(seq[0])):
                if seq[i][j] == 'A':
                    f_mat[0][j] += 1
                elif seq[i][j] == 'C':
                    f_mat[1][j] += 1
                elif seq[i][j] == 'G':
                    f_mat[2][j] += 1
                elif seq[i][j] == 'T':
                    f_mat[3][j] += 1

        total_lenght = len(seq) * len(seq[0])
        for i in range(len(f_mat)):
            for j in range(len(f_mat[0])):
                f_mat[i][j] += (prob[i] * k)


        for i in range(len(f_mat)):
            for j in range(len(f_mat[0])):
                f_mat[i][j] /= (len(seq)+k)
        print("Corrected Frequency Matrix")
        for i in range (len(f_mat)):
            print(f_mat[i])
        return f_mat

    def get_scoring_matrix(seq,prob,k):
        f_mat = PSSM.get_corr_frequency_matrix(seq, prob ,k)
        for i in range(len(f_mat)):
            for j in range(len(f_mat[0])):
                f_mat[i][j] /= prob[i]
        print("Score Matrix")
        for i in range (len(f_mat)):
            print(f_mat[i])
        return f_mat
    """
    def plot_sequence_logo(seq,prob,k):
        colors = {'G': 'orange',
                        'A': 'green',
                        'C': 'blue',
                        'T': 'red'}
        pcolors = list(colors.keys())
        score = PSSM.get_scoring_matrix(seq,prob,k)
    """
