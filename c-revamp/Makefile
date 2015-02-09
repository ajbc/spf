CC = g++ -O3 -larmadillo -lgsl -Wall

LSOURCE = main.cpp utils.cpp data.cpp spf.cpp
CSOURCE = utils.cpp data.cpp


# main model
spf: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf

profile: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf -pg


# comparison methods
pop: popularity.cpp $(CSOURCE)
	  $(CC) popularity.cpp $(CSOURCE) -o pop

mf: mf.cpp $(CSOURCE)
	  $(CC) mf.cpp $(CSOURCE) -o mf


# cleanup
clean:
	-rm -f spf pop mf
