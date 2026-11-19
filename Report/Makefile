################################
################################
##          Makefile          ##
################################
################################
FILENAME = rac
DEPEND = NO
READER = evince
EDITOR = emacs

ifeq ($(DEPEND),DONE)
include $(FILENAME).depend
endif

.PHONY : all clean new view depend again find_dependencies

depend :
	@make find_dependencies
	@make DEPEND=DONE all;
	@echo '';
	@echo '$(FILENAME).pdf up to date';
	@echo '';

all : $(FILENAME).pdf
	@if [ $(DEPEND) != DONE ]; then echo '';\
					echo 'Run "make", not "make all"';\
					echo '';\
					exit 1;\
				   fi

$(FILENAME).pdf : $(FILENAME).tex $(INCLUDES) $(FIGS) $(BIB_FILES)
	@if [ $(DEPEND) != DONE ]; then echo '';\
					echo 'Run "make", not "make $(FILENAME).dvi"';\
					echo '';\
					exit 1;\
				   fi
	@pdflatex $(FILENAME).tex;
	@bibtex $(FILENAME)
	#@makeindex -s nomencl.ist -o $(FILENAME).gls $(FILENAME).glo
	@pdflatex $(FILENAME).tex;
	@if `grep -q -e "There were undefined references" -e "There were multiply-defined labels" -e "Label(s) may have changed. Rerun to get cross-references right" $(FILENAME).log`; \
	  then make DEPEND=DONE again; \
	fi

again :
	@if [ $(MAKELEVEL) = 6 ]; then \
	   echo "Too many nesting levels of make!"; exit 1; \
	 fi
	@if [ $(DEPEND) != DONE ]; then echo '';\
					echo 'Run "make", not "make again"';\
					echo '';\
					exit 1;\
				   fi
	@pdflatex $(FILENAME).tex;
	@if `grep -q -e "There were undefined references" -e "There were multiply-defined labels" -e "Label(s) may have changed. Rerun to get cross-references right" $(FILENAME).log`; \
	  then make DEPEND=DONE again; \
	fi


find_dependencies :
	@echo 'determining dependancies';
	@echo '#dependancies generated automatically with texdepend' > $(FILENAME).depend;
	@echo '#' >> $(FILENAME).depend;
	@perl texdepend $(FILENAME) >> $(FILENAME).depend;
clean :
	rm -f *.aux *.log *~ *.out *.toc *.bbl *.blg *.bak *.depend *.idx *.glo *.gls *.ilg

new : clean
	rm -f *.dvi *.pdf *.ps

view : depend
	$(READER) $(FILENAME).pdf &

wc : 
	@echo "Counting words for all *.tex files in current directory or sub directories."
	@echo "(TeXcount.pl must be in PATH.)"
	@TeXcount.pl `find ./ -name "*.tex"`

edit :
	${EDITOR} ${FILENAME}.tex &
