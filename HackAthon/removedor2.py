one_sents, zero_sents = "",""
lim = 10
c1, c2 = 0, 0

with open("yelp_labelled.txt",'r') as f:
	with open("dest.txt", "w+") as dt:
		for l in f:
			s,t = l.split("\t")
			s += " "
			if t == "0\n":
				zero_sents += s
				if c1 == lim:
					zero_sents += "  \t0\n"
					c1 = 0
				else:
					c1 += 1
			else:
				one_sents += s
				if c2 == lim:
					one_sents += "  \t1\n"
					c2 = 0
				else:
					c2 += 1
		
		dt.write(zero_sents)
		if 0 < c1 < lim:
			dt.write("  \t0\n")
		dt.write(one_sents)
		if 0 < c2 < lim:
			dt.write("  \t1\n")
