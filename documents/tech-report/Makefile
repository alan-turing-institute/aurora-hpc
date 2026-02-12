tech-report-aurora-dawn.pdf: main.tex body.tex styles.tex bibliography.bib
	latexmk -xelatex main.tex ; \
	cp main.pdf tech-report-aurora-dawn.pdf

clean:
	rm -f *.blg *.log *.pdf *.bbl *.aux *.out *.fdb_latexmk *.fls *.bcf *.run.xml *.xdv
