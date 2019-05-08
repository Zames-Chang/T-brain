def score(yhat,y):
    z = 0
    z2 = 0
    total = 0
    for y1,y2 in zip(yhat,y):
        present = abs(y1-y2)/y2
        total += present
        if(present <= 0.1):
            z += 1
        if(present > 0.2):
            z2 += 1
    print('(1-total/len(y)) = ',(1-total/len(y)))
    print(f'> 20 % : {z2/len(y)}')
    return z/len(y),z/len(y)*10000 + (1-total/len(y))