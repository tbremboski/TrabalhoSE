import sys, copy
import numpy as np

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

	# inicializando populacao aleatoriamente
	population = []
	for i in xrange(M):
		population.append(np.random.permutation(N_TASKS))

	# inicio do genetico
	for k in xrange(GEN):
		# selecao de individuos
		pop_sel = []
		for i in xrange(M):
			a = np.random.random_integers(0, M-1)
			b = np.random.random_integers(0, M-1)

			if fitness(population[a]) >= fitness(population[b]):
				pop_sel.append(population[a])
			else:
				pop_sel.append(population[b])

		# crossover
		pop_cross = []
		for i in xrange(0, int(M*PC) - 1, 2):
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
					if pop_sel[i][j] == pop_sel[i+1][j]:
						son[j] = pop_sel[i][j]
					elif j >= a and j <= b:
						son[j] = pop_sel[i][j]
					else:
						pass

				tmp = []
				for j in xrange(N_TASKS):
					if pop_sel[i+1][j] not in son:
						tmp.append(pop_sel[i+1][j])

				c = 0
				for j in xrange(N_TASKS):
					if son[j] == -1:
						son[j] = tmp[c]
						c += 1

				sons.append(son)

			t = []
			for j in range(len(sons)):
				t.append((fitness(sons[j]), j))

			dtype = [('fitness', float), ('index', int)]
			t = np.array(t, dtype=dtype)
			t.sort(order='fitness')

			pop_cross.append(sons[t[-1][1]])
			pop_cross.append(sons[t[-2][1]])

		l = len(pop_cross)
		for i in xrange(M - 1, l - 1, -1):
			pop_cross.append(pop_sel[i])

		# mutacao
		for i in xrange(M):
			r = np.random.random_sample()
			if r <= PM:
				print 'Mutation!!!!'
				a = np.random.random_integers(0, N_TASKS-1)
				b = np.random.random_integers(0, N_TASKS-1)

				tmp = pop_cross[i][a]
				pop_cross[i][a] = pop_cross[i][b]
				pop_cross[i][b] = tmp

		# nova populacao
		population = copy.copy(pop_cross)

		# para printar resultado parcial
		f = []
		for i in xrange(M):
			f.append(fitness(population[i]))

		f = np.array(f)
		f.sort()

		print 'End of generation ' + str(k) + '. Best fitness: ' + str(f[-1])

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