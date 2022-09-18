from numba import cuda, jit, typeof
from numpy import complex64, int64, int32, uint64, float32, float64, complex128,  array, reshape

#2D
def unpack_edges():
    f_edges = open('f4_edges.txt', 'r')
    edges = [int32(x) for x in range(0)]
    l = [line.strip() for line in f_edges]
    for line in l:
        edges.append(list(map(int, line.split())))
    f_edges.close()
    return array(edges)

#3D
def build_chambers(edges):
    chambers = [int32(x) for x in range(0)]
    for i in range(0, 1152):
        chamber = [int32(x) for x in range(0)]
        for j in range(0, 4):
            chamber.append(edges[4 * i + j])
        chambers.append(chamber)
    return array(chambers)

#2D
def unpack_vectors():
    f_vectors = open('f4_socr_nabor.txt', 'r')
    vectors = [int32(x) for x in range(0)]
    char_vectors = [line.strip() for line in f_vectors]
    for line in char_vectors:
        vectors.append(list(map(int, line.split())))
    f_vectors.close()
    return array(vectors)

#1D
def generate_indecies():
    jopak = [int32(x) for x in range(0)]
    u = 1151
    while u > -1:
        jopak.append(u)
        u = u - 1
    return array(jopak)




@cuda.jit('void(int32[:],int32[:, :], int32[:, :, :], int32[:])')
def gpu_happy_cycling(indecies, vectors, chambers, existence):


    thread = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if thread > 1151:
        return
    system_found = False
    quan_sep_systems = 0

    existence = 0

    index = indecies[thread]


    while system_found == False and index > -1:
        j = index - 1
        while j > -1 and system_found == False:
            k = j - 1
            while k > -1 and system_found == False:
                h = k - 1
                while h > - 1 and system_found == False:

                    quan_sep_systems = quan_sep_systems + 1
                    l = 0
                    count = 0
                    forbidden_vector = False
                    while l <= 25612 and system_found == False and forbidden_vector == False:

                        dp1 = 0
                        dp2 = 0
                        dp3 = 0
                        dp4 = 0
                        dp5 = 0
                        dp6 = 0
                        dp7 = 0
                        dp8 = 0
                        dp9 = 0
                        dp10 = 0
                        dp11 = 0
                        dp12 = 0
                        dp13 = 0
                        dp14 = 0
                        dp15 = 0
                        dp16 = 0

                        for i in range(0, 4):
                            dp1 = dp1 + (vectors[l][i])*chambers[index][i][0]
                            dp2 = dp2 + (vectors[l][i])*chambers[index][i][1]
                            dp3 = dp3 + (vectors[l][i])*chambers[index][i][2]
                            dp4 = dp4 + (vectors[l][i])*chambers[index][i][3]
                            dp5 = dp5 + (vectors[l][i])*chambers[j][i][0]
                            dp6 = dp6 + (vectors[l][i])*chambers[j][i][1]
                            dp7 = dp7 + (vectors[l][i])*chambers[j][i][2]
                            dp8 = dp8 + (vectors[l][i])*chambers[j][i][3]
                            dp9 = dp9 + (vectors[l][i])*chambers[k][i][0]
                            dp10 = dp10 + (vectors[l][i])*chambers[k][i][1]
                            dp11 = dp11 + (vectors[l][i])*chambers[k][i][2]
                            dp12 = dp12 + (vectors[l][i])*chambers[k][i][3]
                            dp13 = dp13 + (vectors[l][i])*chambers[h][i][0]
                            dp14 = dp14 + (vectors[l][i])*chambers[h][i][1]
                            dp15 = dp15 + (vectors[l][i])*chambers[h][i][2]
                            dp16 = dp16 + (vectors[l][i])*chambers[h][i][3]

                        spp1 = dp1 > 0
                        spp2 = dp2 > 0
                        spp3 = dp3 > 0
                        spp4 = dp4 > 0
                        spp5 = dp5 > 0
                        spp6 = dp6 > 0
                        spp7 = dp8 > 0
                        spp8 = dp8 > 0
                        spp9 = dp9 > 0
                        spp10 = dp10 > 0
                        spp11 = dp11 > 0
                        spp12 = dp12 > 0
                        spp13 = dp13 > 0
                        spp14 = dp14 > 0
                        spp15 = dp15 > 0
                        spp16 = dp16 > 0

                        if (spp1 and spp2 and spp3 and spp4) or (spp5 and spp6 and spp7 and spp8) or (spp9 and spp10 and spp11 and spp12) or (spp13 and spp14 and spp15 and spp16):
                            count = count + 1
                            l = l + 1
                        else:
                            forbidden_vector = True
                    if count == len(vectors):
                        system_found = True

                    existence = existence + int(system_found)
                    h = h - 1
                k = k -1
            j = j - 1



'''
--------------------------------------------------
Get values
--------------------------------------------------
'''
our_edges = unpack_edges()

our_chambers = build_chambers(our_edges)

first_chamber = our_chambers[0]
other_chambers = our_chambers[1:1152]
del our_chambers

our_vectors = unpack_vectors()

our_indecies = generate_indecies()


'''
--------------------------------------------------
Check values
--------------------------------------------------

print(len(our_edges))
print(our_edges)

print(len(other_chambers))
print(other_chambers)

print(len(first_chamber))
print(first_chamber)
print(type(first_chamber[0]))

print(len(our_vectors))
print(our_vectors)

print(our_indecies)
print(typeof(our_indecies))
print(typeof(our_vectors))
print(typeof(other_chambers))
print(other_chambers[1][1][1])
print(typeof(other_chambers[1][1][1]))
'''
device = cuda.get_current_device()
existence = 0

#Send values to GPU memory
d_indecies = cuda.to_device(our_indecies)
d_vectors = cuda.to_device(our_vectors)
d_chambers = cuda.to_device(other_chambers)
empty_1d = [int32(x) for x in range(1)]
d_existence = cuda.to_device(empty_1d)

#Grid
blocks_per_grid = 32
threads_per_block = 128

#Call kernel
gpu_happy_cycling[blocks_per_grid, threads_per_block](d_indecies, d_vectors, d_chambers, d_existence)

answer = d_existence.copy_to_host()
print(answer)

'''
f_sep_system = open('f4_solution.txt', 'w+')


if the_flag:
    print('Решение найдено! Подходящая система в файле f4_solution.txt')
    the_system = systema
    the_system.insert(0, first_chamber)
    solution = ''

    for chambers in the_system:
        for edge in chambers:
            solution = solution + str(int(edge[0])) + ' ' + str(int(edge[1])) + ' ' + str(int(edge[2])) + ' ' + str(int(edge[3])) + '\n'
    f_sep_system.write(solution)


else:
    print('Подходящей системы не нашлось! Пустое множество в файле f4_solution.txt')
    solution = '[]'
    f_sep_system.write(solution)

f_sep_system.close()
'''