import optimization as op

# s = [1, 4, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]

# op.printschedule(s)
# print(op.schedulecost(s))
domain = [(0, 9)] * (len(op.people) * 2)
# s = op.randomoptimize(domain, op.schedulecost)
# s = op.hillclimb(domain, op.schedulecost)
# s = op.annealingoptimize(domain, op.schedulecost)
# s = op.geneticoptimize(domain, op.schedulecost)
# print(domain,s)
# print(op.schedulecost(s))
# op.printschedule(s)

result = {'rand': [], 'hill': [], 'anne': [], 'gene': []}
for i in range(10):
    s1 = op.randomoptimize(domain, op.schedulecost)
    result['rand'].append(op.schedulecost(s1))

    s2 = op.hillclimb(domain, op.schedulecost)
    result['hill'].append(op.schedulecost(s2))

    s3 = op.annealingoptimize(domain, op.schedulecost)
    result['anne'].append(op.schedulecost(s3))

    s4 = op.geneticoptimize(domain, op.schedulecost)
    result['gene'].append(op.schedulecost(s4))

print(result)
print('rand', sum(result['rand']) / 10, min(result['rand']))
print('hill', sum(result['hill']) / 10, min(result['hill']))
print('anne', sum(result['anne']) / 10, min(result['anne']))
print('gene', sum(result['gene']) / 10, min(result['gene']))
