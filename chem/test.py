from loader import MoleculeDataset
dataset_name = ["tox21", "hiv", "bace", "bbbp", "toxcast", "sider", "clintox"]
num_tasks = [12, 1, 1, 1, 617, 27, 2]
goal_task = [7, 0, 0, 0, 501, 9, 1]
for dataset,num_task in zip(dataset_name, num_tasks):
    print(dataset)

    flag = False 
    if(dataset=='muv'):
        flag=True
    dataset = MoleculeDataset("dataset/"+dataset, dataset = dataset)
    print(dataset) 
    positive = [0 for _ in range(num_task)]
    negitive = [0 for _ in range(num_task)]
    maxx=-1
    minn=0
    goalid = -1 
    for data in dataset:
        for i in range(num_task):
            if(data.y[i].item() == 1):
                positive[i]+=1
            elif(data.y[i].item() == -1):
                negitive[i]+=1 
    answer = [[x,y] for x,y in zip(positive,negitive)]
    idx = 0
    for x,y in answer:
        if(x+y>2*maxx):
            maxx = x+y
            minn = abs(maxx-minn)
            goalid = idx
        elif(x+y>=0.8*maxx and abs(x-y)<minn):
            maxx = x+y
            minn = abs(x-y)
            goalid = idx
        idx+=1    
    print(goalid)    
    print(answer[goalid])

    if(flag):
        print(answer)


