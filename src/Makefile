CC = g++ -O3 -larmadillo -lgsl -Wall -static-libstdc++

LSOURCE = main.cpp utils.cpp data.cpp spf.cpp eval.cpp
CSOURCE = utils.cpp data.cpp eval.cpp


# main model
spf: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf

profile: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf -pg


# comparison methods
pop: popularity.cpp $(CSOURCE)
	  $(CC) popularity.cpp $(CSOURCE) -o pop

rand: random.cpp $(CSOURCE)
	  $(CC) random.cpp $(CSOURCE) -o rand

mf: mf.cpp $(CSOURCE)
	  $(CC) mf.cpp $(CSOURCE) -o mf

librec_eval: librec.cpp $(CSOURCE)
	  $(CC) librec.cpp $(CSOURCE) -o librec_eval


# cleanup
clean:
	-rm -f spf pop mf librec_eval rand
