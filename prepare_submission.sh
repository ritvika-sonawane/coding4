#!/bin/bash

OUTFILE="submit.zip"

# Check presence of hypothesis files
if [ ! -f decoded_hyp_monolingual.txt ]; then
    echo "WARNING: decoded_hyp_monolingual.txt not found."
fi

if [ ! -f decoded_hyp_bilingual.txt ]; then
    echo "WARNING: decoded_hyp_bilingual.txt not found."
fi

# Remove previous zip if exists
if [ -f "$OUTFILE" ]; then
    echo "Removing existing $OUTFILE ..."
    rm "$OUTFILE"
fi

echo "Creating $OUTFILE ..."

zip -r "$OUTFILE" \
    decoded_hyp_monolingual.txt \
    decoded_hyp_bilingual.txt \
    *.py \
    2>/dev/null

echo "Done. Created $OUTFILE."
