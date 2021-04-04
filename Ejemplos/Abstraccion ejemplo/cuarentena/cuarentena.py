class Cuarentena:
    
    def __init__(self):
        pass

    def rutina_diaria(self, flojera = 'poca'):
        self._despertar()
        self._comer()
        self._platzi()
        self._procrastinar(flojera)

    def _despertar(self):
        print('despertar tarde')

    def _comer(self):
        print('alimentarse')

    def _platzi(self):
        print('estudiar con platzi')

    def _procrastinar(self, flojera):
        if flojera == 'poca':
            print('seguir estudiando')
        else:
            print('descansar')

if __name__ == '__main__':
    cuarentena = Cuarentena()
    cuarentena.rutina_diaria()
    cuarentena._procrastinar('mucha')