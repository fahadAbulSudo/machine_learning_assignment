import fastapi
from pydantic import BaseModel
from typing import Optional

app = fastapi.FastAPI()


@app.get("/")
def index():
    """
    Simple page with just localhost / and you will see a dictionary
    :return:
    """
    return {"data": {"name": "Afour"}}


@app.get('/blog')
def index(limit=10, published: bool = True, sort: Optional[str] = None):    # adding query parameters in the URL
    """
    # only get {limit} number of blogs from the list of blogs inside db
    :param limit:
    :param published:
    :return:
    """
    if published:
        return {'data': f'{limit} published blogs from the db'}
    else:
        return {'data': f'{limit} blogs from the db'}


@app.get('/about')   # This is called path
def about():         # This is called path operation functions
    """
    About page information
    :return:
    """
    return {'data': 'about page'}


@app.get('/blog/unpublished')
def unpublished_blogs():
    """
    Show unpublished blogs
    :return:
    """
    return {'data': 'all unpublished blogs'}


@app.get('/blog/{id}')
def show_block(id: int):    # Defining the type of the parameter. id should be int instead of string.
    """
    Fetch blog with id = id
    :param id:
    :return:
    """
    return {'data': id}


@app.get('/blog/{id}/comments')
def comments(id):
    """
    Fetch blog comments with id = id
    :param id:
    :return:
    """
    return {'data': {'1', '2'}}


class Blog(BaseModel):
    """
    This is a class to define a new blog
    """
    id: int
    title: str
    body: str
    published: Optional[bool]


@app.post('/blog')
def add_new_blog(blog: Blog):
    """
    Function to create a post request to add a new blog
    :return:
    """
    return {'data': f'A new blog is added with id: {blog.id}'}
