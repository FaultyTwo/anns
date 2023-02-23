'''
A class for handling with VGG-like chromosomes
'''
import random, misc
from pathlib import Path

class VGGChroClass:
    '''
    Processing chromosome information for VGGLike classes
    Strictly for VGG-like only
    '''
    @classmethod
    def chro_str2chunk(self, gene: str) -> list:
        '''
        Turn a chromosome string into chunk layers (list)

        - gene -> A single chromosome in string format
        '''

        # there's gotta be a better way to seperate the layers than this

        chunk = list()
        cool_str = str()
        #print(gene)
        for x in gene:
            if x.isalpha() == True:
                chunk.append(cool_str)
                cool_str = ""  # purge
            cool_str += x
        chunk.append(cool_str) # append last layer because i suck at coding
        del chunk[0] # seriously, where did this empty element come from?
        # print(chunk)
        return chunk

    @classmethod
    def chro_chunk2str(self, chunk: list):
        '''
        Turn a chunk into a chromosome string

        - chunk -> A single chromosome in string format
        '''
        gene = ""
        for x in chunk:
            gene += str(x)
        return gene

    @classmethod
    def chro_check(self, gene: str):
        '''
        Check the validity of a chromosome

        - gene -> A single chromosome in string format

        If it's valid, return the input
        If it's not valid, return -1 instead
        '''

        # just in case if people keeps forgetting
        if type(gene) is list:
            gene = self.chro_chunk2str(gene)

        path = Path(__file__).parent / "chromo_format/nn_vgg_data.json"

        return misc.chro_check(gene,path)

    @classmethod
    def chro_generate(self, population: int, size, output_channel: int):
        '''
        Generate a set of random chromosomes

        ### Parameters
        - population -> total population of the chromosomes
        - size -> length of all chromosomes, either set (for range) or a single integer
        - output_channel -> desired output channels of all chromosomes

        *Beware that randomly generated chromosomes are unreliable, only use in GA.
        '''

        is_size_range = type(size) is list or type(
            size) is tuple or type(size) is set
        if is_size_range:  # if true
            min, max = int(size[0]), int(size[1] - 1)
        else:
            min, max = int(size - 1), int(size - 1)

        if min < 0:
            raise Exception("Size or min value can't be less than zero!")
        
        if output_channel <= 0:
            raise Exception("Output channels can't be less than one!")

        out = list()
        # generation use mini data
        path = Path(__file__).parent / "chromo_format/nn_vgg_gen_mini_data.json"

        alpha = ['P', 'F', 'C']
        for _ in range(population):  # how many parents?
            is_valid = False
            while is_valid == False:
                chunk = list()
                for _ in range(0, random.randint(min, max)):
                    chunk.append(random.choice(alpha)) # get a string of alphabets
                if misc.chro_check(chunk,path) == -1:
                    continue
                is_valid = True
                chunk = misc.random_layer(chunk,output=output_channel)
                out.append(''.join(chunk))
        return out
