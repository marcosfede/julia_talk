
Dates.today()

f(x) = x * x

code_llvm(f, (Float64,))

code_native(f, (Float64,))

Pkg.add("IJulia")

ccall((:clock, "libc"), Int32, ())

path = ccall((:getenv, "libc"), Cstring, (Cstring,), "SHELL")

unsafe_string(path)

using JavaCall
JavaCall.init(["-Xmx128M"])

jlm = @jimport java.lang.Math
jcall(jlm, "sin", jdouble, (jdouble,), pi/2)

j_u_arrays = @jimport java.util.Arrays
jcall(j_u_arrays, "binarySearch", jint, (Array{jint,1}, jint), [10,20,30,40,50,60], 40)

jcall(JString("hello"), "toUpperCase", JString,())

using PyCall
@pyimport math
math.sin(math.pi / 4) - sin(pi / 4)

@pyimport numpy.random as nr
nr.rand(3,4)

run(`pwd`)

run(`ls`)

x = rand(3,4)

x[1,:]

x[:,1]

x[1,1]

y = x

x[1,:] = zeros(1,4)

x

y

z = x[2,:]

x = zeros(3,4)

z

x

z2 = x[:,:]

x[1,:] = 3

x

z2

x = rand(4,3)

x'

x2 = x * x'

det(x2)

pinv(x2)

row = rand(4)

row'

row' * row

[1 2 3 4]

[1 ; 2 ; 3 ; 4]

[1 2 ; 3 4]

[1, 2, 3, 4]

[[1,2] [3,4]]

[[1,2] ; [3,4]]

sum([[1,2] , [3,4]])

x = rand(4,3)

sum(x)

sum(x,1)

sum(x,2)

x^2

x.^2

x .> 0.5

x[x .> 0.5]

typeof(3)

t = typeof(3)
typeof(t)

x = collect(1:10)

map(e -> e ^ 2, x)

filter(e -> e % 2 == 0, x)

foldl((t,e) -> t + e, 0, x)

intOrString(x::Union{Int, String}) = println(x)

intOrString(3) # this works
intOrString("a") # this works
intOrString(true) # this fails

function sumReal(x::Array{T, N} where T <: Real where N)
    sum(x)
end

sumReal([1,2,3]) # this works
sumReal([1.0,2.0,3.0]) #this works
sumReal([1im, 2im, 3im]) #this fails

t = (2,true)
println(typeof(t))
(a,b) = t
println(a)
println(b)

s = Set([1,2,3])
6 in s

d = Dict(2 => "a", 3 => "b", 4 => "c")
println(typeof(d))
println((2 => "a") in d)
haskey(d, 1)
get(d, 9, "not found")

abstract type Tree{T} end
struct Node{T} <: Tree{T} 
    left::Tree{T}
    right::Tree{T}
end
struct Leaf{T} <: Tree{T}
    value::T
end
struct Empty{T} <: Tree{T} end

#Type parameter can be omitted if it is not necessary
height(t::Node) = max(height(t.left), height(t.right)) + 1
height(t::Leaf) = 0
height(t::Empty) = -1

#Method size from Base has to be explicitly imported in order to be extended.
Base.size(t::Node) = size(t.left) + size(t.right) + 1
Base.size(t::Leaf) = 1
Base.size(t::Empty) = 0

# Note that functions can be declared either in short-form for one-line expressions or in long-form.
show(t::Tree) = show(t, 0)
show(t::Leaf, margin::Int) = println("$(repeat(" ", margin))Leaf($(t.value))")
show(t::Empty, margin::Int) = println("$(repeat(" ", margin))-")
function show(t::Node, margin::Int)
    println("$(repeat(" ", margin))Node")
    show(t.left, margin + 4)
    show(t.right, margin + 4)
end


tree = Node(Node(Leaf(1), Leaf(2)), Node(Leaf(3), Leaf(4)))
println(typeof(tree))
println("height = $(height(tree))")
println("size = $(size(tree))")
show(tree)

struct T
    x::Int
    y::Bool
end

println(T(2, true) === T(2, true))
t = T(2, true)
t = T(3, false) #points to a different object
t.x = 6 # this throws an error

struct T2
    x::Int
    y::Bool
    z::Vector{Int} # an immutable type with a mutable field
end

println(T2(2, true, [1,2,3]) === T2(2, true, [1,2,3])) # this no longer holds because field z is mutable
t = T2(2, true, [1,2,3])
t.z[3] = 9 # the mutable field can be modified
println(t)
t.z = [3,4,5] # but the field cannot point to a different object, because T2 is immutable.

t.z[:] = [3,4,5] # this works because it is modifying the object without changing the reference
t

%%python

macro python_str(s) open(`python`,"w",STDOUT) do io; print(io, s); end; end

python"""
class A:
    def f(self):
        print("A")

class B:
    def f(self):
        print("B")

xs = [A(), B()]
for x in xs:
    x.f()
"""

python"""
class A:
    def f(self):
        print("A")

    def doSomething(self, other):
        other.doSomethingWithA(self)

    def doSomethingWithA(self, other):
        print("AA")

    def doSomethingWithB(self, other):
        print("AB")

class B:
    def f(self):
        print("B")

    def doSomething(self, other):
        other.doSomethingWithB(self)

    def doSomethingWithA(self, other):
        print("BA")

    def doSomethingWithB(self, other):
        print("BB")

xs = [A(), B()]
for x in xs:
    for y in xs:
        x.doSomething(y)
"""

?methods

methods(size)

struct A end
struct B end

doSomething(x::A, y::A) = println("AA")
doSomething(x::A, y::B) = println("AB")
doSomething(x::B, y::A) = println("BA")
doSomething(x::B, y::B) = println("BB")

xs = [A(), B()]
for x in xs
    for y in xs
        doSomething(x,y)
    end
end

function f(a,b)
  return 2*a+b
end

@code_native f(2.0,3.0)

@code_native f(2,3)

function f()
  return 2*a+b
end

a=2
b=3
@code_native f()

a=2.0 
b=3.0
@code_native f()

@code_warntype f()

Pkg.add("RDatasets")
Pkg.add("DataFrames")
using RDatasets, DataFrames

RDatasets.datasets()

size(RDatasets.datasets())

unique(RDatasets.datasets()[:Package])

carDatasets = RDatasets.datasets("car")
carDatasets[map(e -> ismatch(r"^S", e), carDatasets[:Dataset]), :]

df = dataset("car", "Salaries")

size(df)

head(df)

names(df)

describe(df)

df[:Salary]

df[:Salary]

df[:, 6]

df[1, :]

df[:Rank] .== "Prof"

df[df[:Rank] .== "Prof", :]

Pkg.add("DataFramesMeta")
using DataFramesMeta
@where(df, :Rank .== "Prof")

map(n -> (n, eltype(df[n])), names(df))

numCols = filter(n -> eltype(df[n]) <: Number, names(df))

groupby(df, :Rank)

by(df, :Rank, g -> DataFrame(mean = mean(g[:Salary]), min = minimum(g[:Salary]), max = maximum(g[:Salary])))

Pkg.add("Plots")
using Plots

backends()

backend()

plotlyjs()

plot(rand(3,4))

pyplot()

plot(rand(3,4))

names(df)

Pkg.add("StatPlots")
using StatPlots

scatter(df, :YrsSincePhD, :YrsService)

scatter(df, :Rank, :YrsService)

histogram(df, :YrsService, leg=false)

plotlyjs()
counts = by(df, :Rank, g -> DataFrame(count = size(g,1)))
pie(counts[:Rank], counts[:count], axis=false)

counts = by(df, :Discipline, g -> DataFrame(count = size(g,1)))
bar(counts[:Discipline], counts[:count], label="Discipline")

boxplot(["YrsService" "YrsSincePhD"], [df[:YrsService], df[:YrsSincePhD]], leg=false)

series = [@where(df, :Sex.== "Male")[:Salary], 
          @where(df, :Sex.== "Female")[:Salary]]
boxplot(["Male" "Female"], series, leg=false)

using DataFrames

items = DataFrame(profit = [6, 5, 8, 9, 6, 7, 3], weight = [2, 3, 6, 7, 5, 9, 4])

MAX_WEIGHT = 9

module GenAlg
using DataFrames

abstract type CrossOverMethod end
struct UniformCrossOver <: CrossOverMethod end
struct OnePointCrossOver <: CrossOverMethod end
struct TwoPointCrossOver <: CrossOverMethod end

abstract type SelectionMethod end
struct RouletteSelection <: SelectionMethod end
struct TournamentSelection <: SelectionMethod 
    size::Int
end

mutable struct GenAlgConf
    maxGen::Int
    populationSize::Int
    nDims::Int
    nValues::Int
    crossoverProb::Float64
    mutationProb::Float64
    crossoverMethod::CrossOverMethod
    selectionMethod::SelectionMethod
    useElitism::Bool
    fitnessFunction::Function
end

#type aliases, to make code more readable. They have no performance penalty.
const Individual = Vector{Float64}
const Population = Matrix{Float64}
const PopFitness = Vector{Float64}

function initPopulation(conf::GenAlgConf)::Population
    floor.(rand(conf.populationSize, conf.nDims) * conf.nValues)
end

function selectParents(conf::GenAlgConf, pop::Population, popFitness::Vector{Float64}, 
        selectionMethod::RouletteSelection) :: Tuple{Int, Int}
    function getParentIndex(probs::Vector{Float64})::Int
        toss = rand()
        parent = 1
        accProb = probs[parent]
        while (parent < size(probs,1) && accProb < toss) 
            parent += 1 
            accProb += probs[parent]
        end
        parent
    end    
    probs = popFitness ./ sum(popFitness)
    parent1 = getParentIndex(probs)
    probs[parent1] = 0 #prevent it from being chosen again
    parent2 = getParentIndex(probs)
    (parent1, parent2)    
end

function selectParents(conf::GenAlgConf, pop::Population, popFitness::Vector{Float64}, 
        selectionMethod::TournamentSelection) :: Tuple{Int, Int}
    contestants = randperm(conf.populationSize)[1:selectionMethod.size]
    (parent1, _) = selectParents(conf, pop[contestants,:], popFitness[contestants], RouletteSelection())
    contestants = randperm(conf.populationSize)[1:selectionMethod.size]
    (parent2, _) = selectParents(conf, pop[contestants,:], popFitness[contestants], RouletteSelection())
    (parent1, parent2)
end

function mutate!(individual::Individual, conf::GenAlgConf)
    for i in 1:conf.nDims
        if rand() < conf.mutationProb
            individual[i] = floor(rand() * conf.nValues)
        end
    end
end

function crossover(conf::GenAlgConf, parent1::Individual, parent2::Individual, 
        crossOverMethod::UniformCrossOver) :: Tuple{Individual, Individual}
    (child1, child2) = (similar(parent1), similar(parent2))
    for i in 1:conf.nDims
        if rand() < conf.crossoverProb
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        else
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        end
    end
    mutate!(child1, conf)
    mutate!(child2, conf)
    (child1, child2)
end

function crossover(conf::GenAlgConf, parent1::Individual, parent2::Individual, 
        crossOverMethod::OnePointCrossOver) :: Tuple{Individual, Individual}
    (child1, child2) = (similar(parent1), similar(parent2))    
    if rand() < conf.crossoverProb
        # second halves are swapped
        cutPoint = floor(Int, rand() * (conf.nDims - 1))
        child1[1:cutPoint] = parent1[1:cutPoint]
        child2[1:cutPoint] = parent2[1:cutPoint]
        child1[cutPoint+1:end] = parent2[cutPoint+1:end]
        child2[cutPoint+1:end] = parent1[cutPoint+1:end]
    else
        #no crossover, children are copies of the parents
        child1[:] = parent1[:]
        child2[:] = parent2[:]
    end
    mutate!(child1, conf)
    mutate!(child2, conf)
    (child1, child2)
end

function crossover(conf::GenAlgConf, parent1::Individual, parent2::Individual, 
        crossOverMethod::TwoPointCrossOver) :: Tuple{Individual, Individual}
    (child1, child2) = (similar(parent1), similar(parent2))
    cutPoints = sort(randperm(conf.nDims - 1)[1:2])
    if rand() < conf.crossoverProb
        # middle halves are swapped
        child1[1:cutPoints[1]] = parent1[1:cutPoints[1]]
        child1[cutPoints[1]+1:cutPoints[2]] = parent2[cutPoints[1]+1:cutPoints[2]]
        child1[cutPoints[2]+1:end] = parent1[cutPoints[2]+1:end]
        child2[1:cutPoints[1]] = parent2[1:cutPoints[1]]
        child2[cutPoints[1]+1:cutPoints[2]] = parent1[cutPoints[1]+1:cutPoints[2]]
        child2[cutPoints[2]+1:end] = parent2[cutPoints[2]+1:end]
    else
        #no crossover, children are copies of the parents
        child1[:] = parent1[:]
        child2[:] = parent2[:]
    end
    mutate!(child1, conf)
    mutate!(child2, conf)
    (child1, child2)
end

# it is a convention that in-place functions end with "!"
function makeNewPopulation!(conf::GenAlgConf, pop::Population, 
        popFitness::Vector{Float64}, newPop::Population)
    for i in 1:2:conf.populationSize
        (parent1, parent2) = selectParents(conf, pop, popFitness, conf.selectionMethod)
        (newPop[i, :], newPop[i + 1, :]) = 
            crossover(conf, pop[parent1,:], pop[parent2,:], conf.crossoverMethod)
    end
end

function run(conf::GenAlgConf)::Tuple{Population, Individual}
    pop = initPopulation(conf)
    popFitness = conf.fitnessFunction(pop)
    newPop = similar(pop)
    newPopFitness = similar(popFitness)
    bestIndividual = zeros(conf.nDims)
    bestFitness = 0.0
    bestOldIndex = indmax(popFitness)
    for gen in 1:conf.maxGen
        if popFitness[bestOldIndex] > bestFitness
            bestFitness = popFitness[bestOldIndex]
            bestIndividual[:] = pop[bestOldIndex, :]
        end
        makeNewPopulation!(conf, pop, popFitness, newPop)
        newPopFitness[:] = conf.fitnessFunction(newPop)
        if conf.useElitism            
            worstNewIndex = indmin(newPopFitness)
            newPopFitness[worstNewIndex] = popFitness[bestOldIndex]
            newPop[worstNewIndex, :] = pop[bestOldIndex, :]
        end
        pop[:] = newPop[:]
        popFitness[:] = newPopFitness[:]
        bestOldIndex = indmax(popFitness)
    end
    (newPop, bestIndividual)
end

end #module


function getFitnessFunc(items::DataFrame, maxWeight::Int) ::Function
     function fitnessFunc(pop::GenAlg.Population) :: Vector{Float64}
            profit::Vector{Float64} = pop * convert(Vector{Float64}, items[:profit])         
            weight::Vector{Float64} = pop * convert(Vector{Float64}, items[:weight])
            (profit ./ (abs.(weight - maxWeight) + eps()))
        end
    return fitnessFunc
end

MAX_GEN = 100
POP_SIZE = 100
N_DIMS = size(items, 1)
N_VALUES = 2 # binary coding
P_CROSS = 0.8
P_MUT = 0.001
USE_ELITISM = true
CROSSOVER = GenAlg.UniformCrossOver()
#CROSSOVER = GenAlg.OnePointCrossOver()
#CROSSOVER = GenAlg.TwoPointCrossOver()
SELECTION = GenAlg.RouletteSelection()
#SELECTION = GenAlg.TournamentSelection(5)
fitness = getFitnessFunc(items, MAX_WEIGHT)

@code_warntype(fitness(rand(10, size(items, 1))))

conf = GenAlg.GenAlgConf(MAX_GEN, POP_SIZE, N_DIMS, N_VALUES, P_CROSS, P_MUT, 
    CROSSOVER, SELECTION, USE_ELITISM, fitness)

(newPopulation, bestIndividual) = GenAlg.run(conf)

newFitness = conf.fitnessFunction(newPopulation)

mean(newFitness)

minimum(newFitness)

maximum(newFitness)

bestIndividual

bestIndividual' * items[:profit]
