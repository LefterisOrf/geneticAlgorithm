import numpy as np

class GeneticAlgorithm():
    """
        A class that solves a one dimensional problem using a genetic algorithm.
        Takes as arguments on creation the following:
        - populationSize: The population number (e.g. 100 possible sollutions in each gen)
        - geneSize: The number of genes each possible sollution has
        - generations: The max number of generations to run. Given as argument in run()
        - mask: a list containing only 0's and 1's which define the parent gene
          that will pass to the child. len(mask) must equal the populationSize.
        - scoringFunction: A scoring function, which returns 1 when a possible sollution is correct
        - mutationRate: a float in the range [0, 1) which defines how many child genes
          will be randomly changed
        - minVal: the minimum value that a gene might take
        - maxVal: the maximum value that a gene might take
        ########################################################################
        REQUIREMENTS:
            - numpy
            - python3
            - a scoring function which takes one possible sollution and evaluates it
              must return 1 on best sollution. For all other possible sollutions the score
              must be in range (0, 1).
    """
    def __init__(self, populationSize, geneSize, mask, scoringFunction, mutationRate, minVal = 0, maxVal = 200):
        self.populationSize = populationSize
        self.geneSize = geneSize
        self.mask = mask
        self.scoringFunction = scoringFunction
        self.mutationRate = mutationRate
        self.minVal = minVal
        self.maxVal = maxVal
        if minVal == 0.0 and maxVal == 1.0:
            self.genFunction = np.random.uniform
        else:
            self.genFunction = np.random.randint

    def run(self, maxGenerations = 12000):
        sollutionFound = False
        population = []
        generation = 0
        np.random.seed(None)
        # Initialize the first population and calculate each candidate's score
        for i in range(self.populationSize):
            candidate = self.genFunction(self.minVal, self.maxVal, size = self.geneSize)
            population.append(candidate)

        while (not sollutionFound) and (generation < maxGenerations):
            print("In generation: ", generation)
            scores = []
            for i in range(self.populationSize):
                score = self.scoringFunction(population[i])
                if score == 1:
                    print("Sollution found. Candidate: ", population[i])
                    return population[i]
                scores.append(score)
            """
             Calculate the propability of each candidate to become a parent
             depending on its score
            """
            scoreSum = sum(scores)
            # print("scoreSum: ", scoreSum)
            propabilities = []
            for i in range(self.populationSize):
                if i == 0:
                    propabilities.append(scores[i]/scoreSum)
                else:
                    propabilities.append(scores[i]/scoreSum + propabilities[i - 1])
            #print(propabilities[len(propabilities) - 1])
            # Generate the new population
            newPopulation = []
            generation += 1
            #print("In parent selection:")
            for i in range(self.populationSize // 2):
                # Take two random numbers at [0, 1]
                m = np.random.rand()
                mParent = -1
                nParent = -1
                n = np.random.rand()
                # print("m: ", m," n: ",n)
                # Determine to which candidate each number coresponds to
                for c in range(self.populationSize):
                    if m <= propabilities[c] and mParent == -1:
                        mParent = c
                    if n <= propabilities[c] and nParent == -1:
                        nParent = c
                    if nParent != -1 and mParent != -1:
                        break
                if mParent == -1 or nParent == -1:
                    print("Parent not found correctly.")
                    return None
                #print(" Parents selected: ", mParent, nParent)
                # Create the child from m and n, depending on the mask
                child1 = []
                child2 = []
                for c in range(self.geneSize):
                    if self.mask[c] == 0:
                        child1.append(population[mParent][c])
                        child2.append(population[nParent][c])
                    else:
                        child1.append(population[nParent][c])
                        child2.append(population[mParent][c])
                newPopulation.append(np.array(child1))
                newPopulation.append(np.array(child2))
            population = newPopulation
            newPopulation = []
            attributeCount = self.populationSize * self.geneSize
            for c in range(int(attributeCount * self.mutationRate)):
                index = np.random.randint(0, self.populationSize)
                candIndex = np.random.randint(0, self.geneSize)
                value = self.genFunction(self.minVal, self.maxVal)
                # change a random number of a candidate
                population[index][candIndex] = value
            # New population is finished. Now we need to repeat the process
        print("Failed to find an optimal sollution")
        # Find best available sollution
        finalScores = []
        for i in population:
            finalScores.append(self.scoringFunction(i))
        indexMax = np.argmax(finalScores)
        bestSollution = population[indexMax]
        print("Best available sollution: ", bestSollution)
        print("Best sollution's result: ", bestSollution[0] + 2 * bestSollution[1] + 3 * bestSollution[2] + 4 * bestSollution[3])
def errorCalc(candidate):
    # calculate the |f(a,b,c,d)|
    f = abs(candidate[0] + 2 * candidate[1] + 3 * candidate[2] + 4 * candidate[3] - 9)
    return 1 / (1 + f)

def main():
    genAlg = GeneticAlgorithm(100, 4, [1, 0, 1, 0], errorCalc, 0.1, 0.0, 1.0)
    genAlg.run()

if __name__ == '__main__':
    main()
