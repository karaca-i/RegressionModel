import matplotlib.pyplot as plt

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    dlblue = '#0096ff'
    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False
    
def plot_classification(x_train, y_train, X_train2, y_train2):
    pos = y_train == 1
    neg = y_train == 0

    dlblue = '#0096ff'
    fig,ax = plt.subplots(1,2,figsize=(8,3))

    #plot 1, single variable
    ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlblue,lw=3)

    ax[0].set_ylim(-0.08,1.1)
    ax[0].set_ylabel('y', fontsize=12)
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].set_title('one variable plot')
    ax[0].legend()

    #plot 2, two variables
    plot_data(X_train2, y_train2, ax[1])
    ax[1].axis([0, 4, 0, 4])
    ax[1].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_xlabel('$x_0$', fontsize=12)
    ax[1].set_title('two variable plot')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    