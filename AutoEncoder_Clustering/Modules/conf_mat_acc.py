from mlxtend.plotting import plot_confusion_matrix

def conf_mat_acc(confusion_matrix_tensor,
                 num_labels,
                 len_data):
    total = 0
    for i in range(len(confusion_matrix_tensor)):
        total += confusion_matrix_tensor[i][i]
    
    acc = (total/len_data) * 100
    return acc