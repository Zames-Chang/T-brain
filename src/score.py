def score(y_true, y_predict):
    z = 0
    z2 = 0
    total = 0
    for y1, y2 in zip(y_predict, y_true):
        present = abs(y1 - y2)/abs(y2)
        total += present
        if(present <= 0.1):
            z += 1
        if(present > 0.2):
            z2 += 1
    return z/len(y_true)*10000 + (1-total/len(y_true))