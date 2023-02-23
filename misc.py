'''
Miscellaneous methods.
Helper functions. Shared by classes.
'''

from automata.fa.dfa import DFA
import json, random

def chro_check(gene:list,path):
        '''
        Backend method for checking chromosome

        Used in every chromosomes checking methods.
        '''
        gene = ''.join(gene)

        with open(path,"r") as f:
            dfa_struct = json.load(f)
        
        dfa = DFA(
            states=set(dfa_struct['states']),
            final_states=set(dfa_struct['accept']),
            input_symbols=set(dfa_struct['alphabet']),
            initial_state=dfa_struct["initial_state"],
            transitions=dfa_struct["trans"],
            allow_partial=True
        )
        if dfa.accepts_input(gene):
            return gene
        else:
            return -1 # this gene sucks, go build it again or something

def random_layer(layers, max_kernel:int = 5, max_neurons:int = 100, output = None):
    '''
    Helper method for generating a layer

    Limited to Convolution, Pooling, and Linear layers.
    '''
    l_layers = list()
    if type(layers) is str:
        #print("is string")
        for x in layers:
            l_layers.append(x)
    elif type(layers) is list:
        l_layers = layers

    for x in range(len(l_layers)):
        if l_layers[x] == 'C':
            ran_val = str(random.randint(1, max_kernel) * 11)
        elif layers[x] == 'P': # 1x1 pooling does nothing but eat hot chips and lie
            ran_val = str(random.randint(2, max_kernel) * 11)
        else:
            ran_val = str(random.randint(10, max_neurons))
        l_layers[x] += ran_val

    if output != None:
        l_layers.append('F' + str(output))
    else:
        pass
    
    return l_layers