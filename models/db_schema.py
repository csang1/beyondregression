db.define_table(
    'uploaded_content',
    Field('filename','upload'),
    Field('number_rows','integer',writable=False,readable=False),
    Field('column_names','list:string',writable=False,readable=False),
    auth.signature)
