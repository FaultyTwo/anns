'''
Automatic Neural Network class.
This class handles the entire pipeline.
'''

import random, time, warnings, math

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.grad_scaler as amp
import matplotlib.pyplot as plt

class AutoNN:
    '''
    Automatic Neural Network
    '''
    def __formatize(self,l:list): # turn one dim to two dim with error rate of -1
        '''
        Turn on dim to two dim with error rate of -1

        Probably will vectorize this if parents is too large
        '''
        out = list()
        for x in l:
            out.append([x])
        for y in range(0,len(out)):
            out[y].append(-1)
        return out

    def trainandtest(self,child:list,n_epochs:int,train_data,test_data,
    learning_rate:float,model:object,
    interval_rate:int,device):
        '''
        Train and test a list of chromosomes

        ### Parameters
        - child -> A list of chromosomes for evaluation in GA
        - n_epochs -> Training Epoches for each neural network training
        - train_data -> Training data in torch.utils.data.DataLoader object
        - test_data -> Testing data in torch.utils.data.DataLoader object
        - learning_rate -> Learning rate of optimizer
        Default optimizer is ADAM
        - model -> a model type to create
        - log_interval -> Console log interval
        - device -> Device for processing neural network
        '''
        n = model()
        log_interval = interval_rate

        dat = enumerate(train_data)
        _, (data_shape, _) = next(dat)
        dvc_type = "cpu" if device == "cpu" else "cuda"
        dtype = torch.bfloat16 if device == "cpu" else torch.float16

        for x in range(0,len(child)):
            chro = child[x][0] # get the child chromosome
            model = n.model_build(chro,list(data_shape.shape)) # build the model
            model = model.to(torch.device(device)) # casting to gpu or cpu
            # model = torch.compile(model) # for pytorch 2.0
            # TODO: Allow user to specify optimizer and loss function
            # Optimizer shouldn't be problem. But loss function, well, should be
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scaler = amp.GradScaler() if device != "cpu" else None
            
            def train(epochs):
                print("\n== Training Session ==")
                model.train() # tell the model we are training
                for batch_idx, (data, target) in enumerate(train_data):
                    #print(batch_idx,data,target,log_interval)

                    optimizer.zero_grad() # zero gradient
                    data = data.to(torch.device(device))
                    target = target.to(torch.device(device)) # casting data to whatever

                    with torch.autocast(device_type=dvc_type,dtype=dtype):
                        output = model(data) # fit the data
                        loss = F.nll_loss(output, target, reduction='mean')

                    # scaler forwarding for cuda devices
                    if device != "cpu":
                        scaler.scale(loss).backward() # backward
                        scaler.step(optimizer) # optimizer step
                        scaler.update() # update the scale
                    else:
                        loss.backward()
                        optimizer.step() # OPTIMIZE IT YOU DORK

                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_data.dataset),
                        100. * batch_idx / len(train_data), loss.item()))
                # out of loop
                print("Epoch:",epochs,"\tAvg. Training Loss:",round(loss.item(),4))
            
            def test():
                print("\n== Testing Session ==")
                model.eval() # tell the model we are evaluating
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in test_data:
                        data = data.to(torch.device(device))
                        target = target.to(torch.device(device)) # casting data to whatever
                        output = model(data)
                        test_loss += F.nll_loss(output, target, reduction='mean').item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_data.dataset)
                print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_data.dataset),
                100. * correct / len(test_data.dataset)))
                return float(100. * correct / len(test_data.dataset))
            
            print("\nBegin the evaluation for ",chro,"\nplease wait a decade or so...")
            tic = time.perf_counter()
            try:
                for epoch in range(1, n_epochs + 1):
                    train(epoch) # train
                t_loss = test() # test for last, speed up, jeez
            except KeyboardInterrupt:
                raise Exception("Aborted. No graph for you!")
            except:
                print("Found an error while training/test. assigning accuracy of 0.0 ...")
                t_loss = 0.0
            toc = time.perf_counter()
            print("Accuracy for",chro,":",round(t_loss,4))
            print("Train/test time for",chro,":", round(toc - tic,4),"seconds")

            # maybe in the future, we should allow people to save model
            # rather choosing the best one then
            model, optimizer, scaler = None, None, None # flush previous parameters
            child[x][1] = t_loss

        return child

    def random_error(self,l:list): # randomize the error, for testing purpose
        '''
        Randomize the dummy error for each genes, for testing purpose
        '''
        for x in range(0,len(l)):
            l[x][1] = (round(random.random()*100,3))
        return l

    def GeneticAlgorithm(self,ga_iter:int,parents:list,train_data,
    test_data,model,crossover,mutation
    ,n_epochs:int = 1,learning_rate:float=0.001
    ,log_interval=256,device='cpu',plot_name=None):
        '''
        Evaluate a list of parents chromosome of a dataset
        
        ### Parameters
        - ga_iter -> Numbers of iterations for Genetic Algorithm evaluation
        - parents -> A list of parents chromosomes for crossover and evaluate
        - train_data -> Training data in torch.utils.data.DataLoader object
        - test_data -> Testing data in torch.utils.data.DataLoader object
        - model -> types of model to create and evaluate, must support the chromosome
        model => (chro, input_shape)
        - crossover -> prefer crossover method, must support the chromosome
        crossover => (list of chro)
        - mutation -> prefer mutation method, must support the chromosome
        crossover => (list of chro, greedy selection)
        - n_epochs -> Training Epoches for each neural network training
        - learning_rate -> Learning rate of SGD optimizers
        - log_interval -> Console log interval
        - device -> Device for processing neural network
        - plot_name -> Plot the result, if None -> won't plot * REMOVE THIS LATER
        '''
        # replace random_error function with pytorch error when using for real
        # otherwise, IT'S REAL! THE DREAM!
        # add parents validity checking here before continue
        if len(parents) < 4:
            raise Exception("Can't perform GA with parents less than four")

        print("== GA Report ==")
        print("Parents:",parents)
        print("GA Iteration:",ga_iter)
        print("Train/Test Epoches:",n_epochs)
        print("Learning Rate:",learning_rate)
        print("Crossover:",crossover.__name__)
        print("Mutation:",mutation.__name__)
        
        plot_x,plot_y = list(),list()

        parents = self.__formatize(parents) # to two dim
        parents = self.trainandtest(parents,n_epochs,train_data,test_data,
        learning_rate,model,log_interval,device)
        #parents = self.random_error(parents)
        parents.sort(key= lambda k: k[1], reverse=True)
        plot_y.append(self.__plot_formatize(parents))
        plot_x.append(1)

        print("\nGeneration 1:")
        print(parents)

        chro_only = list()
        # ok, this next part is going to be important
        # i need to sort based on train and test score somehow\
        for x in range(ga_iter):
            for y in range(0,len(parents)):
                chro_only.append(parents[y][0]) # get chromosome only, not the error rate
            child = crossover(chro_only) # crossover overloaded
            child = mutation(child,math.floor(len(child)/2)) # might need to work on greedy functionality
            #child = mutation(child) # might need to work on greedy functionality
            child = self.__formatize(child)
            child = self.trainandtest(child,n_epochs,train_data,test_data,
            learning_rate,model,log_interval,device)
            # elitism survivors
            family = parents + child
            family.sort(key= lambda k: k[1], reverse=True)
            parents = family[:len(parents)]
            plot_y.append(self.__plot_formatize(parents))
            plot_x.append(x+2)
            chro_only = [] # oops. forgot to purge chro_only
            print("\nGeneration:", x+2)
            print(parents)
        
        try:
            if plot_name != None and type(plot_name) is str:
                plt.figure(1)
                plt.plot(plot_x,plot_y)
                plt.title("Generations/Test Accuracy (%)")
                plt.ylabel("Test Accuracy (%)")
                plt.xlabel("Generations")
                plt.grid(True)
                plt.savefig(plot_name)
        except:
            print("Can't plot the graph for some strange reason idc")
        
        return parents

    def __plot_formatize(self,parents:list):
        ret = list()
        for x in range(0,len(parents)):
            ret.append(parents[x][1])
        return ret
