def create_classes(db):
    class Stock(db.Model):
        __tablename__ = 'RealTimeStock'

        id = db.Column(db.Integer, primary_key=True)
        date = db.Column(db.Date(64))
        open = db.Column(db.Float(25))
        high = db.Column(db.Float(25))
        low = db.Column(db.Float(25))
        close = db.Column(db.Float(25)) 
        adjclose = db.Column(db.Float(25)) 
        volume = db.Column(db.Float(25)) 
        ticker = db.Column(db.String(25))

        def __repr__(self):
            return '<Stock %r>' % (self.id)
    return Stock
