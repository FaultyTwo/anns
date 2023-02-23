import random, misc
from pathlib import Path

class InceptionChroClass:
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
        sep = gene.split('I')
        sep = ' '.join(sep).split() # die empty members, die
        #print(sep)
        for index,x in enumerate(sep):
            if x[0] == 'S': # inception part
                inception_list = x.split('S')
                del inception_list[0] # remove empty member, ''
                #print(inception_list)
                sep[index] = inception_list
        return sep

    @classmethod
    def chro_chunk2str(self, chunk: list):
        '''
        Turn a chunk into a chromosome string

        - chunk -> A single chromosome in string format
        '''
        pass

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

        path = Path(__file__).parent / "have fun writing dfa for this chromosome lmao"

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
        pass