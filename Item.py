import pandas


class Item:
        
    def __init__(self, store_id, item_id):
        self.store_id = store_id
        self.item_id = item_id
        self.dataset = pandas.DataFrame()
        
        
    def selectFrom(self, dataset):
        self.dataset = dataset.loc[(dataset.store == self.store_id) & (dataset.item == self.item_id), ['sales']]
        self.dataset = self.dataset.asfreq('D')