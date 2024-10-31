import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from cvxopt import matrix as cvx_matrix, solvers as cvx_solvers
import matplotlib
import io
import base64

matplotlib.use('Agg')  

def plot_hyperplanes(request):
    x_hard = np.array([[1., 3.], [2., 2.], [1., 1.]])
    y_hard = np.array([1, 1, -1])

    n = x_hard.shape[0]
    H = cvx_matrix(np.outer(y_hard, y_hard) * np.dot(x_hard, x_hard.T))
    q = cvx_matrix(-np.ones((n, 1)))
    G = cvx_matrix(-np.eye(n))
    h = cvx_matrix(np.zeros(n))
    A = cvx_matrix(y_hard.reshape(1, -1).astype(float))
    b = cvx_matrix(np.zeros(1))

    sol_hard = cvx_solvers.qp(H, q, G, h, A, b, kktsolver='ldl')
    lamb_hard = np.array(sol_hard['x'])
    w_hard = np.sum(lamb_hard[i] * y_hard[i] * x_hard[i] for i in range(n)).flatten()
    
    sv_idx = np.where(lamb_hard > 1e-5)[0]
    sv_x_hard = x_hard[sv_idx]
    sv_y_hard = y_hard[sv_idx].reshape(1, -1)

    plt.figure(figsize=(5, 5))
    color_hard = ['red' if a == 1 else 'blue' for a in y_hard]
    plt.scatter(x_hard[:, 0], x_hard[:, 1], s=200, c=color_hard, alpha=0.7)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    
    b_hard = np.mean(sv_y_hard - np.dot(sv_x_hard, w_hard))
    x1_dec = np.linspace(0, 4, 100)
    x2_dec = - (w_hard[0] * x1_dec + b_hard) / w_hard[1]
    plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

    w_norm_hard = np.sqrt(np.sum(w_hard ** 2))
    half_margin_hard = 1 / w_norm_hard
    upper_hard = x2_dec + 2 * half_margin_hard * (w_hard[1] / w_norm_hard)
    lower_hard = x2_dec - 2 * half_margin_hard * (w_hard[1] / w_norm_hard)
    
    plt.plot(x1_dec, upper_hard, '--', lw=1.0, label='positive boundary')
    plt.plot(x1_dec, lower_hard, '--', lw=1.0, label='negative boundary')

    plt.scatter(x_hard[sv_idx, 0], x_hard[sv_idx, 1], s=50, marker='o', c='white')

    for s, (x1, x2) in zip(lamb_hard, x_hard):
        plt.annotate('λ=' + str(s[0].round(2)), (x1-0.05, x2 + 0.2))

    plt.legend()
    plt.title("\nMargin = {:.4f}".format(half_margin_hard * 2))
    buf_hard = io.BytesIO()
    plt.savefig(buf_hard, format='png')
    plt.close()
    buf_hard.seek(0)

    x_soft = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694],
                       [0.176, 0.458], [0.306, 0.753], [0.936, 0.413], [0.215, 0.410],
                       [0.612, 0.375], [0.784, 0.602], [0.612, 0.554], [0.357, 0.254],
                       [0.204, 0.775], [0.512, 0.745], [0.498, 0.287], [0.251, 0.557],
                       [0.502, 0.523], [0.119, 0.687], [0.495, 0.924], [0.612, 0.851]])
    y_soft = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1])
    y_soft = y_soft.astype('float').reshape(-1, 1)

    C = 50.0
    N = x_soft.shape[0]
    H_soft = np.dot((y_soft * x_soft), (y_soft * x_soft).T)
    P = cvx_matrix(H_soft)
    q = cvx_matrix(np.ones(N) * -1)

    A = cvx_matrix(y_soft.reshape(1, -1))
    b = cvx_matrix(np.zeros(1))

    g = np.vstack([-np.eye(N), np.eye(N)])
    G = cvx_matrix(g)
    h1 = np.hstack([np.zeros(N), np.ones(N) * C])
    h = cvx_matrix(h1)

    sol_soft = cvx_solvers.qp(P, q, G, h, A, b, kktsolver='ldl')    
    lamb_soft = np.array(sol_soft['x'])
    w_soft = np.sum(lamb_soft * y_soft.flatten().reshape(-1, 1) * x_soft, axis=0)
    
    sv_idx_soft = np.where(lamb_soft > 1e-5)[0]
    lamb_soft[sv_idx_soft]
    sv_x_soft = x_soft[sv_idx_soft]
    sv_y_soft = y_soft[sv_idx_soft]

    plt.figure(figsize=(7, 7))
    color_soft = ['red' if a == 1 else 'blue' for a in y_soft.flatten()]
    plt.scatter(x_soft[:, 0], x_soft[:, 1], s=200, c=color_soft, alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    b_soft = sv_y_soft[0] - np.dot(sv_x_soft[0], w_soft)
    x1_dec_soft = np.linspace(0, 1, 100)
    x2_dec_soft = -(w_soft[0] * x1_dec_soft + b_soft) / w_soft[1]
    plt.plot(x1_dec_soft, x2_dec_soft, c='black', lw=1.0, label='decision boundary')

    y_hat = np.dot(x_soft, w_soft) + b_soft
    slack = np.maximum(0, 1 - y_soft.flatten() * y_hat)
    for s, (x1, x2) in zip(slack, x_soft):
        plt.annotate(str(s.round(2)), (x1-0.02, x2 + 0.02))
    w_norm = np.linalg.norm(w_soft)
    margin_soft = 1 / w_norm
    w_unit_soft = w_soft / w_norm
    upper_soft = np.vstack([x1_dec_soft, x2_dec_soft]).T + margin_soft * w_unit_soft
    lower_soft = np.vstack([x1_dec_soft, x2_dec_soft]).T - margin_soft * w_unit_soft


    plt.plot(upper_soft[:, 0], upper_soft[:, 1], '--', lw=1.0, label='positive boundary')
    plt.plot(lower_soft[:, 0], lower_soft[:, 1], '--', lw=1.0, label='negative boundary')
    plt.scatter(sv_x_soft[:, 0], sv_x_soft[:, 1], s=60, marker='o', c='white')
    plt.legend()
    plt.title('C = ' + str(C) + ',  Σξ = ' + str(np.sum(slack).round(2)))
    buf_soft = io.BytesIO()
    plt.savefig(buf_soft, format='png')
    plt.close()
    buf_soft.seek(0)
    
    return render(request, 'index.html', {
        'hard_plot': base64.b64encode(buf_hard.getvalue()).decode(),
        'soft_plot': base64.b64encode(buf_soft.getvalue()).decode(),
    })