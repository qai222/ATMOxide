"""
CSD-system (aka CSD python API) and CSD-WEB (CSD-WEB, the website you go for searching and downloading cif files directly) are built based on two similar but different databases.
The cif files in CSD-WEB are the "deposition records", they are (almost) identical to the cif files crystallographers uploaded, it is the "raw" file.
The cif files in CSD-system are the so-called curated ones: No diffraction info, and the fields are "standardized".
The consequence is for the same CSD identifier, some fields present in CSD-WEB.cif could be absent in CSD-system.cif. Here is my conversation from ccdc using QOXCEN as an example:
    In my research, the "QOXCEN-WEB.cif" is much preferred as it contains detailed information about disorder. My question is, how can I get the deposition CIF like "QOXCEN-WEB.cif" from CSD-API?
Unfortunately this isn't possible because the web database contains more detailed data than the database accessed by the CSD software including the Python API, which only contains the data which our software can make use of - if we tried to put all the data from the web database into the software database it would be even larger than it is now.
This may change in the future but for the moment the detailed deposition CIF file can only be retrieved from the web.
"""
