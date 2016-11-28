import hashlib 
from mathmodels import *

def index():
    return dict()

@auth.requires_login()
def first():
    form = SQLFORM(db.uploaded_content).process()
    if form.accepted:
        filename = os.path.join(request.folder,'uploads',form.vars.filename)
        column_names, number_rows = get_info(filename)
        db(db.uploaded_content.id==form.vars.id).update(column_names=column_names,
                                                        number_rows=number_rows)
        redirect(URL('second', args=form.vars.id))
    return dict(form=form)

@auth.requires_login()
def second():
    uploaded_content = db.uploaded_content(request.args(0,cast=int),created_by=auth.user.id)
    fullname = os.path.join(request.folder,'uploads',uploaded_content.filename)
    rows = uploaded_content.column_names
    length = uploaded_content.number_rows
    # TODO: there is a problem here in case the name contains special symbols
    fields = [Field(row,'boolean',default=False) for row in rows]
    form = SQLFORM.factory(*fields)
    if form.process().accepted: 
        for row in rows:
            if row in request.vars:
                session.choose = row
        redirect(URL('second_half',args=uploaded_content.id))
    elif form.errors:
        response.flash = 'form has errors'
    else:
        response.flash = 'please fill out the form'
    return dict(form=form, length=length)

@auth.requires_login()
def second_half():
    uploaded_content = db.uploaded_content(request.args(0,cast=int),created_by=auth.user.id)
    fullname = os.path.join(request.folder,'uploads',uploaded_content.filename)
    rows = uploaded_content.column_names
    choose = session.choose
    # TODO: there is a problem here in case the name contains special symbols
    field = [Field(row,'boolean',default=False) for row in rows]
    form_delete = SQLFORM.factory(*field)
    delete = []
    if form_delete.process().accepted: 
        for row in rows:
            if row in request.vars:
                delete.append(row)
        redirect(URL('third',args=uploaded_content.id,vars=dict(choose=choose, delete = delete)))
    elif form_delete.errors:
        response.flash = 'form has errors'
    else:
        response.flash = 'please fill out the form'
    return dict(form=form_delete)

@auth.requires_login()
def third():
    uploaded_content = db.uploaded_content(request.args(0,cast=int),created_by=auth.user.id)
    fullname = os.path.join(request.folder,'uploads',uploaded_content.filename)
    return run_models(fullname, request.vars.choose, request.vars.delete)

@auth.requires_login()
def fourth():
    uploaded_content = db.uploaded_content(request.args(0,cast=int),created_by=auth.user.id)
    fullname = os.path.join(request.folder,'uploads',uploaded_content.filename)
    return make_plots(fullname, request.vars.choice)

def user():
    return dict(form=auth())
