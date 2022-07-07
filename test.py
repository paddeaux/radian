
points = 657
vor = 6
groups = []

if points % vor != 0:
    temp = round(points / vor)
    for i in range(vor-1):
        groups.append(temp)
    groups.append(points - sum(groups))
else:
    for i in range(vor):
        groups.append(round(points/vor))

print(groups)
print(sum(groups))
print(round(points / vor))
