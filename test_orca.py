import orca
import pandas as pd

@orca.table('my_table')
def my_table():
    return pd.DataFrame({'a': [1, 2, 3]})

@orca.column('my_table', cache = True, cache_scope = "step")
def b(my_table):
    return my_table.a * 10

@orca.step('output')
def output():
    print orca.get_table("my_table").b
    
@orca.step('remove_b')
def remove_b():
    df = orca.get_table("my_table").to_frame(['a'])
    orca.add_table('my_table', df)
    
@orca.step('make_b_local')
def make_b_local():
    df = orca.get_table("my_table").to_frame(['a', 'b'])
    df['b'] = df['b'] * 10
    orca.add_table('my_table', df)
    
@orca.step('add_b')
def add_b():
    remove_b()
    orca.add_column("my_table", "b", pd.Series([1000, 2000, 3000]), cache_scope = "step")

@orca.step('end')
def end():
    print "The END"

orca.run([
    "output",
    #"make_b_local",
    #"output",
    "remove_b",
    "output",
    "add_b",
    "output",
    "end"
    ], iter_vars=range(1, 4))



