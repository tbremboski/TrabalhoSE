import sys
import numpy as np

def main(argv):
	data = np.loadtxt(argv[0], delimiter=" ")
	for i in range(5):
		media_fit = 0.0
		media_nt = 0.0
		for j in range(10):
			media_fit += data[i*10+j][0]
			media_nt += data[i*10+j][1]
		media_fit /= 10.0
		media_nt /= 10.0

		print str(media_fit) + ' ' + str(media_nt)

if __name__ == '__main__':
	main(sys.argv[1:])