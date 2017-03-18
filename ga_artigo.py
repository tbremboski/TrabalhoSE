import sys, copy
import numpy as np
import random

W = 10				# para uso na geracao de populacao inicial com estrategia FS
M = 100				# tamanho da populacao
T = 250				# periodo de tempo
N_TASKS = 50		# numero de tarefas
GEN = 100			# numero de geracoes
PC = 0.8			# probabilidade de crossover
PM = 0.0006			# probabilidade de mutacao
PC_LOOP = 10		# numero de filhos por crossover (escolhe 2 melhores)
tasks = []			# lista de tarefas global

def fitness(chro):
	# funcao de fitness, conforme o artigo
	xi = np.zeros(N_TASKS, dtype=np.int)
	current_time = 0.0
	last_i = -1
	result = 0.0
	tasks_local = copy.copy(tasks)
	
	for i in xrange(len(chro)):
		task = tasks_local[chro[i]]
		constr1 = False
		constr2 = False
		constr3 = False
		tei = task.tsi

		if current_time > T:
			break

		if current_time <= task.tesi:
			current_time = task.tesi
		if current_time <= task.tlsi:
			task.tsi = current_time
			tei = task.tsi + task.ti

		if 0.0 <= task.tsi and task.tsi < tei and tei <= T:
			constr1 = True

		if task.tesi <= task.tsi and task.tsi <= task.tlsi:
			constr2 = True

		if last_i > -1:
			last_task = tasks_local[last_i]
			if last_task.tsi + last_task.ti <= task.tsi:
				constr3 = True
		else:
			constr3 = True

		if constr1 and constr2 and constr3:
			xi[chro[i]] = 1
			last_i = chro[i]

	for i in xrange(len(chro)):
		result += xi[chro[i]] * tasks_local[chro[i]].ri

	return result

def greed_rule_1():
	c = sorted(tasks, key=lambda x: x.tesi, reverse=False)
	labels = [x.label for x in c]
	return labels

def greed_rule_2():
	c = sorted(tasks, key=lambda x: x.ri, reverse=True)
	labels = [x.label for x in c]
	return labels

def greed_rule_3():
	c = sorted(tasks, key=lambda x: (x.ri / x.ti), reverse=True)
	labels = [x.label for x in c]
	return labels

def greed_rule_4():
	c = sorted(tasks, key=lambda x: x.ti, reverse=False)
	labels = [x.label for x in c]
	return labels

def greed_rule_5():
	c = sorted(tasks, key=lambda x: x.tvi, reverse=False)
	labels = [x.label for x in c]
	return labels

def greed_rule_6():
	c = sorted(tasks, key=lambda x: (x.tvi - x.ti), reverse=False)
	labels = [x.label for x in c]
	return labels

def gera_populacao_fs():
	pop = []
	pop_tmp = []
	bests = []

	for i in xrange(W*M):
		pop_tmp.append(np.random.permutation(N_TASKS))

	for i in xrange(W*M):
		bests.append((fitness(pop_tmp[i]), i))

	dtype = [('fitness', float), ('index', int)]
	bests = np.array(bests, dtype=dtype)
	bests[::-1].sort(order='fitness')

	for i in xrange(0, W*M, 3):
		pop.append(pop_tmp[bests[i][1]])


	return pop


def gera_populacao_hrhs():
	pop = []
	half_pop = []

	n_half_pop = M/2

	for i in xrange(n_half_pop):
		half_pop.append(np.random.permutation(N_TASKS))

	pop.extend(half_pop)



	pop_from_rules = []
	pop_from_rules.append(greed_rule_1())
	pop_from_rules.append(greed_rule_2())
	pop_from_rules.append(greed_rule_3())
	pop_from_rules.append(greed_rule_4())
	pop_from_rules.append(greed_rule_5())
	pop_from_rules.append(greed_rule_6())

	total_fitness = 0.0

	# Soma todas as avaliacoes para uma variavel soma
	for i in range(len(pop_from_rules)):
		total_fitness += fitness(pop_from_rules[i])

	# STEP 4
	for i in xrange(M - n_half_pop):

		s_new = []

		# STEP 3
		while len(s_new) < N_TASKS:

			# STEP 1
			# Selecione um numero s entre 0 e soma
			s = random.uniform(0.0, total_fitness)
			ind = 0
			aux = fitness(pop_from_rules[ind])

			while aux < s:
				ind += 1
				aux += fitness(pop_from_rules[ind])

			# STEP 2
			for ii in xrange(N_TASKS):
				if pop_from_rules[ind][ii] not in s_new:
					s_new.append(pop_from_rules[ind][ii])
					break

			pop.append(s_new)
		
	return pop


def gera_populacao_aleatoria_sem_repeticao():
	pop = []
	all_fit = []
	for i in xrange(M):
		repeat = True
		while repeat:
			chromo = np.random.permutation(N_TASKS)
			f = fitness(chromo)
			if f not in all_fit:
				pop.append(chromo)
				all_fit.append(f)
				repeat = False

	return pop

def gera_populacao_aleatoria():
	pop = []
	for i in xrange(M):
		pop.append(np.random.permutation(N_TASKS))

	return pop

def selecao_roleta(population):
	pop_sel = []
	total_fitness = 0.0

	# Soma todas as avaliacoes para uma variavel soma
	for i in range(M):
		total_fitness += fitness(population[i])

	# Faco isso para toda uma populacao nova
	for i in range(M):

		# Selecione um numero s entre 0 e soma (nao inclusos)
		s = random.uniform(0.0, total_fitness)
		ind = 0
		aux = fitness(population[ind])

		while aux < s:
			ind += 1
			aux += fitness(population[ind])

		pop_sel.append(population[ind])

	return pop_sel

def selecao_aleatoria(population):
	pop_sel = []
	for i in xrange(M):
		a = np.random.random_integers(0, M-1)
		b = np.random.random_integers(0, M-1)

		if fitness(population[a]) >= fitness(population[b]):
			pop_sel.append(population[a])
		else:
			pop_sel.append(population[b])

	return pop_sel

#by Ulisses

def crossover_go_away(population):
	pop_cross = crossover_same_sit(population)
	for task in xrange(0,M,3):
		ind = pop_cross[task]
		pop_cross[task] = ind[::-1]
	return pop_cross

def crossover_random_walk(population,generation):
	from math import floor
	import copy as cp
	pop_cross = crossover_same_sit(population)
	alpha = int(floor(0.3*(M/(generation+1))))
	crazy_walkers = random.sample(xrange(M), alpha)
	for task in crazy_walkers:
		ind = pop_cross[task]
		pop_cross[task] = ind[::-1]
	crazy_swipe = random.sample(xrange(M), alpha)
	for task in crazy_swipe:
		ind = cp.deepcopy(pop_cross[task])
		swipe_pos = random.sample(xrange(N_TASKS), 2)
		aux = ind[swipe_pos[0]] 
		ind[swipe_pos[0]] = ind[swipe_pos[1]]
		ind[swipe_pos[1]] = aux
		pop_cross[task] = ind
	return pop_cross

def crossover_reverse_worst(population):
	pop_cross = crossover_random_walk(population)

	fits = []
	for i,task in enumerate(pop_cross):
		fit = fitness(task)
		fits.append([i,fit])
	sorted_fits = sorted(fits, key = lambda x: int(x[1]),reverse=True)
	for worst in sorted_fits[0:10]:
		ind = pop_cross[worst[0]] 
		pop_cross[worst[0]] = ind[::-1]
	return pop_cross

def crossover_same_sit(population):
	pop_cross = []
	for i in xrange(0, M, 2):
		r = np.random.random_sample()
		if i+1 >= M:
			pop_cross.append(population[i])
		elif r < PC:
			sons = []
			for ii in xrange(PC_LOOP):
				son = np.zeros(N_TASKS, dtype=np.int)
				son -= 1
				a = np.random.random_integers(0, N_TASKS-1)
				b = np.random.random_integers(0, N_TASKS-1)

				if a > b:
					tmp = a
					a = b
					b = tmp

				for j in xrange(N_TASKS):
					if population[i][j] == population[i+1][j]:
						son[j] = population[i][j]
					elif j >= a and j <= b:
						son[j] = population[i][j]
					else:
						pass

				tmp = []
				for j in xrange(N_TASKS):
					if population[i+1][j] not in son:
						tmp.append(population[i+1][j])

				c = 0
				try:
					for j in xrange(N_TASKS):
						if son[j] == -1:
							son[j] = tmp[c]
							c += 1
				except IndexError:
					print "aquele erro nao parou ..."

				sons.append(son)

			t = []
			for j in range(len(sons)):
				t.append((fitness(sons[j]), j))

			dtype = [('fitness', float), ('index', int)]
			t = np.array(t, dtype=dtype)
			t.sort(order='fitness')

			pop_cross.append(sons[t[-1][1]])
			pop_cross.append(sons[t[-2][1]])
		else:
			pop_cross.append(population[i])
			pop_cross.append(population[i+1])

	return pop_cross

def mutacao(population):
	pop = population
	for i in xrange(M):
		r = np.random.random_sample()
		if r < PM:
			print 'Mutation!!!!'
			a = np.random.random_integers(0, N_TASKS-1)
			b = np.random.random_integers(0, N_TASKS-1)

			tmp = pop[i][a]
			pop[i][a] = pop[i][b]
			pop[i][b] = tmp

	return pop

def print_parcial(population, k):
	f = []
	for i in xrange(M):
		f.append(fitness(population[i]))

	f = np.array(f)
	f.sort()

	print 'End of generation ' + str(k) + '. Best fitness: ' + str(f[-1])

def main(argv):
	# inicializando valores de cada tarefa
	v_tesi = np.random.uniform(0, T, N_TASKS)
	v_tesi.sort()
	v_ti = np.random.normal(6, 1, N_TASKS)
	v_ti = np.absolute(v_ti)
	v_tlsi = np.random.normal(5, 1, N_TASKS)
	v_tlsi = np.absolute(v_tlsi)
	v_ri = np.random.uniform(0, 1, N_TASKS)

	# criando lista de tarefas
	for i in xrange(N_TASKS):
		tasks.append(Task(i, v_ti[i], v_tesi[i], v_tlsi[i], v_ri[i]))

	greed_rule_1()
	# inicializando populacao aleatoriamente
	# population = gera_populacao_aleatoria()
	population = gera_populacao_hrhs()
	# population = gera_populacao_fs()
	champions_history = []
	# inicio do genetico
	for k in xrange(GEN):
		# selecao de individuos
		# pop_sel = selecao_aleatoria(population)
		pop_sel = selecao_roleta(population)


		# crossover
		#pop_cross = crossover_same_sit(pop_sel)
		#pop_cross = crossover_go_away(pop_sel)
		pop_cross = crossover_random_walk(pop_sel,k)
		#pop_cross = crossover_reverse_worst(pop_sel)
		# mutacao
		pop_mut = mutacao(pop_cross)

		# nova populacao
		population = copy.copy(pop_mut)

		# para printar resultado parcial
		print_parcial(population, k)
		#se cair num plato obriga a mutar! 
			# ordena resultados pelo fitness
		bests = []
		for i in xrange(M):
			bests.append((fitness(population[i]), i))

		dtype = [('fitness', float), ('index', int)]
		bests = np.array(bests, dtype=dtype)
		bests.sort(order='fitness')
		champions_history.append(bests[-1][0])
		try:
			plato = (champions_history[-1]-champions_history[-2]) - (champions_history[-2]-champions_history[-3])
		except Exception as e:
			plato = 0
		if plato > 1e-2:
			print "plato!!! -> put a little spicy"
			PM = 1
		else:
			PM = 0.0006
	# ordena resultados pelo fitness
	bests = []
	for i in xrange(M):
		bests.append((fitness(population[i]), i))

	dtype = [('fitness', float), ('index', int)]
	bests = np.array(bests, dtype=dtype)
	bests.sort(order='fitness')

	# melhor resultado
	print 'Best fitness: ' + str(bests[-1][0])
	print 'Best task order:'
	print population[bests[-1][1]]


class Task:
	def __init__(self, label, ti, tesi, tlsi, ri):
		self.label = label
		self.ti = ti
		self.tesi = tesi
		self.tlsi = self.tesi + tlsi
		self.tlei = self.tlsi + self.ti
		self.tvi = self.tlei - self.tesi
		self.tsi = -1
		self.ri = ri

if __name__ == '__main__':
	main(sys.argv[1:])