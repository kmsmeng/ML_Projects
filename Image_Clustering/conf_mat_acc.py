def conf_mat_acc(confusion_matrix_tensor, num_labels, len_data):
    '''This functions gives out the accuracy by seeing all the correct labels in the '''
    total = []
    for i in range(num_labels):
        total.append(confusion_matrix_tensor[i][i])
    
    total_sum = sum(total)

    accuracy = (total_sum/len_data) * 100

    return accuracy.item()
