
points = 10000
vor = 6
groups = []

if points % vor != 0:
    temp = round(points / vor)
    for i in range(vor):
        if sum(groups) + temp < points:
            groups.append(temp)
        else:
            groups.append(points - sum(groups))
else:
    for i in range(vor):
        groups.append(round(points/vor))

print(groups)
print(sum(groups))
print(round(points / vor))
