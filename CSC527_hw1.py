# Han Nguyen
# CSC 527 Homework 1


# Prediction function ----------------------------------------------------------------------
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1 if activation >= 0 else 0


# Create AND truth table ----------------------------------------------------------------------
def and_table(n, data, temp):
    if (n == 0):
        tb = int(all(temp))  # true if all elements == 1
        temp.append(tb)
        data.append(temp)
        return
    for i in range(2):
        temp.append(i)
        and_table(n - 1, data, list.copy(temp))
        temp.pop()


# Create OR truth table ----------------------------------------------------------------------
def or_table(n, data, temp):
    if (n == 0):
        tb = int(any(temp))  # true if any elements == 1
        temp.append(tb)
        data.append(temp)
        return
    for i in range(2):
        temp.append(i)
        or_table(n - 1, data, list.copy(temp))
        temp.pop()


# input signal
n = 5

data = []

"""
# AND logic function
and_table(n, data, [])
bk = -5
weights = [bk]+[1]*n
"""

# OR logic function
or_table(n, data, [])
bk = -1
weights = [bk]+[1]*n


print("The truth table/input data: \n", data)
print()

# Testing
for row in data:
    testing = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (row[-1], testing))


# question d
# As the number of input signals increases, the bias value has also be increased and vice versa
