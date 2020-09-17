# Han Nguyen
# CSC 527 Homework 1 - question b


# Prediction function ----------------------------------------------------------------------
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1 if activation >= 0 else 0


# Create AND truth table ----------------------------------------------------------------------
def and_table(n, data1, temp):
    if (n == 0):
        tb = int(all(temp))  # true if all elements == 1
        temp.append(tb)
        data1.append(temp)
        return
    for i in range(2):
        temp.append(i)
        and_table(n - 1, data1, list.copy(temp))
        temp.pop()


# Create OR truth table ----------------------------------------------------------------------
def or_table(n, data2, temp):
    if (n == 0):
        tb = int(any(temp))  # true if any elements == 1
        temp.append(tb)
        data2.append(temp)
        return
    for i in range(2):
        temp.append(i)
        or_table(n - 1, data2, list.copy(temp))
        temp.pop()


# input size
n = 5

data1 = []
# OR logic function
or_table(n, data1, [])
bk1 = -1
weights1 = [bk1]+[1]*n

print("The truth table/input data for OR logic function: \n", data1)
print()
# Testing
for row in data1:
    testing = predict(row, weights1)
    print("Expected=%d, Predicted=%d" % (row[-1], testing))
print()

data2 = []
# AND logic function
and_table(n, data2, [])
bk2 = -5
weights2 = [bk2]+[1]*n

print("The truth table/input data for AND logic function: \n", data2)
print()
# Testing
for row in data2:
    testing = predict(row, weights2)
    print("Expected=%d, Predicted=%d" % (row[-1], testing))