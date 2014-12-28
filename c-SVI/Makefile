CC = g++ -O1 -larmadillo -lgsl

LSOURCE = main.cpp utils.cpp data.cpp spf.cpp

spf: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf

profile: $(LSOURCE)
	  $(CC) $(LSOURCE) -o spf -pg

clean:
	-rm -f example spf
