import sys, copy, csv
import numpy as np
import random
import math

B = 0.1				# taxa melhores
W = 10				# para uso na geracao de populacao inicial com estrategia FS
M = 100				# tamanho da populacao
N_TASKS = 10		# numero de tarefas
GEN = 50			# numero de geracoes
PC = 0.8			# probabilidade de crossover
PM = 0.006			# probabilidade de mutacao
PC_LOOP = 30		# numero de filhos por crossover (escolhe 2 melhores)
# T = math.pow(math.log(math.pow(N_TASKS,6.0), math.e),2)+400	# periodo de tempo
# T = N_TASKS * 6.0 * (10 - (N_TASKS / 10))	# periodo de tempo
# T = 1000
tasks = []			# lista de tarefas global

def fitness(chro):
	# funcao de fitness, conforme o artigo
	num_task = 0
	xi = np.zeros(N_TASKS, dtype=np.int)
	current_time = 0.0
	last_i = -1
	result = 0.0
	tasks_local = copy.copy(tasks)
	old_time = current_time

	for i in xrange(len(chro)):
		task = tasks_local[chro[i]]
		constr1 = False
		constr2 = False
		constr3 = False
		tei = task.tsi

		if current_time > T:
			break

		old_time = current_time
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
			num_task += 1
			current_time = tei
		else:
			current_time = old_time

	for i in xrange(len(chro)):
		result += xi[chro[i]] * tasks_local[chro[i]].ri

	return result, num_task

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
		bests.append((fitness(pop_tmp[i])[0], i))

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
		total_fitness += fitness(pop_from_rules[i])[0]

	# STEP 4
	for i in xrange(M - n_half_pop):

		s_new = []

		# STEP 3
		while len(s_new) < N_TASKS:

			# STEP 1
			# Selecione um numero s entre 0 e soma
			s = random.uniform(0.0, total_fitness)
			ind = 0
			aux = fitness(pop_from_rules[ind])[0]

			while aux < s:
				ind += 1
				aux += fitness(pop_from_rules[ind])[0]

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
			f = fitness(chromo)[0]
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
	pop = []
	pop_sel = []
	total_fitness = 0.0

	# Soma todas as avaliacoes para uma variavel soma
	for i in range(M):
		total_fitness += fitness(population[i])[0]
		pop.append((fitness(population[i])[0], i))

	# Faco isso para toda uma populacao nova
	for i in range(M - int(M*B)):

		# Selecione um numero s entre 0 e soma (nao inclusos)
		s = random.uniform(0.0, total_fitness)
		ind = 0
		aux = fitness(population[ind])[0]

		while aux < s:
			ind += 1
			aux += fitness(population[ind])[0]

		pop_sel.append(population[ind])

	dtype = [('fitness', float), ('index', int)]
	pop = np.array(pop, dtype=dtype)
	pop.sort(order='fitness')

	last = int(M*B)
	last *= -1
	c = pop[last:]

	pop_pass = []
	for e in c:
		# print 'e: ' + str(e[1])
		pop_sel.append(population[e[1]])
		pop_pass.append(population[e[1]])

	return (pop_sel, pop_pass)

def selecao_aleatoria(population):
	# pop = []
	pop_sel = []

	# for i in range(M):
	# 	pop.append((fitness(population[i])[0], i))

	# for i in xrange(M - int(M*B)):
	for i in xrange(M):
		a = np.random.random_integers(0, M-1)
		b = np.random.random_integers(0, M-1)

		if fitness(population[a])[0] >= fitness(population[b])[0]:
			pop_sel.append(population[a])
		else:
			pop_sel.append(population[b])

	# dtype = [('fitness', float), ('index', int)]
	# pop = np.array(pop, dtype=dtype)
	# pop.sort(order='fitness')

	# last = int(M*B)
	# last *= -1
	# c = pop[last:]

	# pop_pass = []
	# for e in c:
	# 	pop_sel.append(population[e[1]])
	# 	pop_pass.append(population[e[1]])

	# return (pop_sel, pop_pass)
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
	pop_cross = crossover_same_sit(population)

	fits = []
	for i,task in enumerate(pop_cross):
		fit = fitness(task)[0]
		fits.append([i,fit])
	sorted_fits = sorted(fits, key = lambda x: x[1],reverse=False)
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
				t.append((fitness(sons[j])[0], j))

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
	pop = copy.copy(population)
	for i in xrange(M):
		r = np.random.random_sample()
		if r < PM:
			# print 'Mutation!!!!'
			a = np.random.random_integers(0, N_TASKS-1)
			b = np.random.random_integers(0, N_TASKS-1)

			tmp = pop[i][a]
			pop[i][a] = pop[i][b]
			pop[i][b] = tmp

	return pop

def print_parcial(population, k):
	f = []
	for i in xrange(M):
		f.append(fitness(population[i])[0])

	f = np.array(f)
	f.sort()

	print 'End of generation ' + str(k) + '. Best fitness: ' + str(f[-1])

def main(argv):
	global N_TASKS
	global T
	n_iter = 0
	ler_arquivo = False
	save = False
	if len(argv) > 0:
		try:
			n_iter = int(argv[0])
			save = True
		except ValueError:
			ler_arquivo = True

	if len(argv) > 1:
		N_TASKS = int(argv[1])
		T = int((math.log(math.pow(N_TASKS, 6.0), math.e)) * 50)

	if ler_arquivo:
		v_ti = []
		v_tesi = []
		v_tlsi = []
		v_ri = []
		data = np.loadtxt(argv[0], delimiter=",")

		for row in data:
			v_ti.append(row[0])
			v_tesi.append(row[1])
			v_tlsi.append(row[2])
			v_ri.append(row[3])

	else:
		# inicializando valores de cada tarefa
		v_tesi = np.random.uniform(0, T, N_TASKS)
		v_tesi.sort()
		v_ti = np.random.normal(6, 1, N_TASKS)
		v_ti = np.absolute(v_ti)
		v_tlsi = np.random.normal(5, 1, N_TASKS)
		v_tlsi = np.absolute(v_tlsi)
		v_ri = np.random.uniform(0, 1, N_TASKS)

	if save:
		rows = []
		for i in range(N_TASKS):
			row = []
			row.append(v_ti[i])
			row.append(v_tesi[i])
			row.append(v_tlsi[i])
			row.append(v_ri[i])
			rows.append(row)
		f_name = 'test-' + str(N_TASKS) + '-' + str(n_iter) + '-adaptativo-new.csv'
		with open(f_name, 'wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerows(rows)

		sys.exit(0)

	# criando lista de tarefas
	for i in xrange(N_TASKS):
		tasks.append(Task(i, v_ti[i], v_tesi[i], v_tlsi[i], v_ri[i]))

	# inicializando populacao aleatoriamente
	# population = gera_populacao_aleatoria()
	# population = gera_populacao_aleatoria_sem_repeticao()
	population = gera_populacao_hrhs()
	# population = gera_populacao_fs()

	# champions_history = []

	# inicio do genetico
	for k in xrange(GEN):
		# selecao de individuos
		pop_sel = selecao_aleatoria(population)
		# pop_sel, pop_pass = selecao_roleta(population)

		# crossover
		pop_cross = crossover_same_sit(pop_sel)
		# pop_cross = crossover_go_away(pop_sel)
		# pop_cross = crossover_random_walk(pop_sel,k)
		# pop_cross = crossover_reverse_worst(pop_sel)

		# mutacao
		pop_mut = mutacao(pop_cross)

		# for i in range(len(pop_pass)):
		# 	r = np.random.random_integers(0, M-1)
		# 	pop_mut[r] = pop_pass[i]

		# nova populacao
		population = copy.copy(pop_mut)

		# para printar resultado parcial
		# print_parcial(population, k)

		#se cair num plato obriga a mutar!
		# ordena resultados pelo fitness
		# bests = []
		# for i in xrange(M):
		# 	bests.append((fitness(population[i]), i))

		# dtype = [('fitness', float), ('index', int)]
		# bests = np.array(bests, dtype=dtype)
		# bests.sort(order='fitness')
		# champions_history.append(bests[-1][0])
		# try:
		# 	plato = (champions_history[-1]-champions_history[-2]) - (champions_history[-2]-champions_history[-3])
		# except Exception as e:
		# 	plato = 0
		# if plato > 1e-2:
		# 	print "plato!!! -> put a little spicy"
		# 	PM = 1
		# else:
		# 	PM = 0.006

	# ordena resultados pelo fitness
	bests = []
	for i in xrange(M):
		fit, nt = fitness(population[i])
		bests.append((fit, i, nt))

	dtype = [('fitness', float), ('index', int), ('nt', int)]
	bests = np.array(bests, dtype=dtype)
	bests.sort(order='fitness')

	# melhor resultado
	# print 'Best fitness: ' + str(bests[-1][0])
	# print 'Best task order:'
	# print population[bests[-1][1]]

	print str(bests[-1][0]) + ' ' + str(bests[-1][2])


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
