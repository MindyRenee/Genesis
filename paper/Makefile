# Build the Genesis white paper.
# Requires a LaTeX toolchain (pdflatex + charter/microtype/booktabs,
# all in texlive-latex-recommended / psnfss).

PDF = genesis.pdf
SRC = genesis.tex

all: $(PDF)

$(PDF): $(SRC)
	pdflatex -interaction=nonstopmode $(SRC)
	pdflatex -interaction=nonstopmode $(SRC)

clean:
	rm -f genesis.aux genesis.log genesis.out genesis.toc

.PHONY: all clean
