# manga-text-localization

OCRs text.  Intended for use with manga pages, but could probably be extended to other uses.

Note this is not complete!  As such, it won't work really well.

**Note**: This was initially made for <https://guya.moe>, and due to this tool being a bit of a pain to use, I settled on working with Azure via [KaguyaOCR](https://github.com/Pyreko/KaguyaOCR) instead.  See that instead ~~though that's also going to get deprecated for another tool I'm working on~~.

## To use:

1.  Create a venv
2.  Make, which will download the pip reqs.
3.  Run using ``python3 image_localization.py`` or just ``./image_localization.py``.  Look at the command line options for what you can pass in. 
