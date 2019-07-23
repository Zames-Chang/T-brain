def score(y_true,y_predict):
    z = 0
    z2 = 0
    total = 0
    for y1,y2 in zip(y_predict,y_true):
        present = abs(y1-y2)/abs(y2)
        total += present
        if(present <= 0.1):
            z += 1
        if(present > 0.2):
            z2 += 1
    print(f"first part : {z/len(y_true)*10000} last part: {1-total/len(y_true)}")
    print(f'> 20 % : {z2/len(y_true)}')
    return z/len(y_true),z/len(y_true)*10000 + (1-total/len(y_true))