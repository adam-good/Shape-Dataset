import numpy as np
import pandas as pd
import argparse
from PIL import Image, ImageDraw
from numpy import cos, sin, sqrt, pi
from numpy.random import randint
from pathlib import Path

np.random.seed(8675309)

def image_shape(s):
    """Acts as a data type so image width/height can be entered on the command line

    Arguments:
        s {string} -- A string containing the width and height of the image, seperated by a comma. 'width,height'

    Raises:
        argparse.ArgumentTypeError: Error raised if input string is not the right format

    Returns:
        tuple(int,int) -- tuple of integers corresponding to width and height
    """
    try:
        w,h = map(int, s.split(','))
        return (w,h)
    except:
        raise argparse.ArgumentTypeError("Shape must be (width,height)")

class DataGenerator():
    def __init__(self, root_dir, img_shape, dataset_size):
        self.root_dir = root_dir
        self.img_shape = self.width, self.height = img_shape
        self.n_items = dataset_size

        self.paths = {
            'circle': f'{self.root_dir}/circles',
            'rectangle': f'{self.root_dir}/rectangles',
            'triangle': f'{self.root_dir}/triangles',
            'obj1': f'{self.root_dir}/obj1s',
            'obj2': f'{self.root_dir}/obj2s'
        }

    def generate_dataset(self):
        for path in self.paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        self._gen_circles(num_items)
        self._gen_triangles(num_items)
        self._gen_rectangles(num_items)
        self._gen_obj1s(num_items)
        self._gen_obj2s(num_items)

        self._gen_label_csv(num_items, num_items, num_items, num_items, num_items)

    def _rotate_point(self, point, origin, angle):
        # Convert angle from degrees to radians (and fix orientation)
        angle = -angle * pi/180

        R = np.array([
            [cos(angle), -sin(angle)],
            [sin(angle), cos(angle)]
        ])
        point = np.array(point)
        origin = np.array(origin)
        point = point - origin
        x,y = R.dot(point) + origin
        return (x,y)

    def _draw_circle(self, image, center, radius):
        cx,cy = center
        lx,ly = cx-radius, cy-radius
        rx,ry = cx+radius, cy+radius
        
        draw = ImageDraw.Draw(image)
        draw.ellipse([(lx,ly), (rx,ry)], fill='white', outline='black')

    def _draw_triangle(self, image, v1,v2,v3 ):
        draw = ImageDraw.Draw(image)
        draw.polygon((v1,v2,v3), fill='white', outline='black')

    def _draw_rectangle(self, image, center, width, height, rotation):
        draw = ImageDraw.Draw(image)
        x,y = center
        v1 = (x-width//2, y-height//2)
        v2 = (x-width//2, y+height//2)
        v3 = (x+width//2, y+height//2)
        v4 = (x+width//2, y-height//2)

        v1 = self._rotate_point(v1, center, rotation)
        v2 = self._rotate_point(v2, center, rotation)
        v3 = self._rotate_point(v3, center, rotation)
        v4 = self._rotate_point(v4, center, rotation)

        draw.polygon([v1,v2,v3,v4], fill='white', outline='black')

    def _draw_equallateral_triangle(self, image, center, edge_len, rotation):
        cx,cy = center
        height = sqrt(edge_len**2 - (0.5*edge_len)**2 )
        ex1, ey1 = (cx, cy - height // 2)
        ex2, ey2 = (cx+edge_len//2, cy + height // 2)
        ex3, ey3 = (cx-edge_len//2, cy+height//2)

        ex1, ey1 = self._rotate_point( (ex1, ey1), center, rotation)
        ex2, ey2 = self._rotate_point( (ex2, ey2), center, rotation)
        ex3, ey3 = self._rotate_point( (ex3, ey3), center, rotation)

        self._draw_triangle(image, (ex1,ey1), (ex2, ey2), (ex3, ey3))

    def _draw_obj1(self, image, center, height, width, radii, rotation):
        ''' Draw a car kinda thing? '''
        x,y = center

        c1 = (x - width//2 + radii//2, y)
        c2 = (x + width//2 - radii//2, y)
        c1 = self._rotate_point(c1, center, rotation)
        c2 = self._rotate_point(c2, center, rotation)
        self._draw_circle(image, c1, radii)
        self._draw_circle(image, c2, radii)

        rec_height = height - 2*radii
        c3 = (x, y - radii - rec_height//2)
        c3 = self._rotate_point(c3, center,rotation)
        self._draw_rectangle(image, c3, width, height-2*radii, rotation)

    def _draw_obj2(self, image, center, edge_len, radii, rotation):
        x,y = center

        c1 = (
                x + (edge_len/sqrt(3) + radii) * cos(90*pi/180),
                y - (edge_len/sqrt(3) + radii) * sin(90*pi/180)
            )
        c2 = (
                x + (edge_len/sqrt(3) + radii) * cos(210*pi/180),
                y - (edge_len/sqrt(3) + radii) * sin(210*pi/180)
            )
        c3 = (
                x + (edge_len/sqrt(3) + radii) * cos(330*pi/180),
                y - (edge_len/sqrt(3) + radii) * sin(330*pi/180)
            )
        c1 = self._rotate_point(c1, center, rotation)
        c2 = self._rotate_point(c2, center, rotation)
        c3 = self._rotate_point(c3, center, rotation)
        
        self._draw_equallateral_triangle(image, center, edge_len, rotation)
        self._draw_circle(image, c1, radii)
        self._draw_circle(image, c2, radii)
        self._draw_circle(image, c3, radii)

    def _gen_circles(self, number):
        dest = self.paths['circle']
        for i in range(number):
            img = Image.new('L', self.img_shape)
            radius = randint(5, min(self.width, self.height)//4)
            center = (
                randint(0+radius, self.width-radius),
                randint(0+radius, self.height-radius)
            )
            self._draw_circle(img, center, radius)
            img.save(f'{dest}/circle_{i}.png', 'PNG')

    def _gen_triangles(self, number):
        dest = self.paths['triangle']
        for i in range(number):
            img = Image.new('L', self.img_shape)
            edge_length = randint(10, min(self.width, self.height)//2)
            center = (
                randint(0+edge_length, self.width-edge_length),
                randint(0+edge_length, self.height-edge_length)
            )
            angle = randint(1, 360)
            self._draw_equallateral_triangle(img, center, edge_length, angle)

            img.save(f'{dest}/triangle_{i}.png', 'PNG')

    def _gen_rectangles(self, number):
        dest = self.paths['rectangle']
        for i in range(number):
            img = Image.new('L', self.img_shape)
            width = randint(10, min(self.width, self.height)//2)
            height = randint(10, min(self.width, self.height)//2)
            center = (
                randint(0+width, self.width-width),
                randint(0+height, self.height-height)
            )
            angle = randint(1, 360)
            self._draw_rectangle(img, center, width, height, angle)

            img.save(f'{dest}/rectangle_{i}.png', 'PNG')

    def _gen_obj1s(self, number):
        dest = self.paths['obj1']
        for i in range(number):
            img = Image.new('L', self.img_shape)
            height = randint(10, min(self.width, self.height)//2)
            width =  randint(height, min(self.width, self.height)//2)
            radii = min(width//3, height//3)
            angle = randint(1,360)
            center = (
                randint(0+width, self.width-width),
                randint(0+height, self.height-height)
            )
            self._draw_obj1(img, center, height, width, radii, angle)
            img.save(f'{dest}/obj1_{i}.png','PNG')

    def _gen_obj2s(self, number):
        dest = self.paths['obj2']
        for i in range(number):
            img = Image.new('L', self.img_shape)
            edge_len = randint(10, min(self.width, self.height)//2)
            radii = randint(3, edge_len//2)
            angle = randint(0, 360)
            
            width = ( sqrt(edge_len**2 - (0.5*edge_len)**2) )

            center = (
                randint(0+width, self.width-width),
                randint(0+width, self.height-width)
            )

            self._draw_obj2(img, center, edge_len, radii, angle)
            img.save(f'{dest}/obj2_{i}.png', 'PNG')

    def _gen_label_csv(self, n_circles, n_rectangles, n_triangles, n_obj1s, n_obj2s):

        n_rows = n_circles + n_rectangles + n_triangles + n_obj1s + n_obj2s

        label_dict = {
            'circle': [1,0,0],
            'rectangle': [0,1,0],
            'triangle': [0,0,1],
            'obj1': [1,1,0],
            'obj2': [1,0,1]
        }

        shapes = np.concatenate([
            np.repeat('circle', n_circles),
            np.repeat('rectangle', n_rectangles),
            np.repeat('triangle', n_triangles),
            np.repeat('obj1', n_obj1s),
            np.repeat('obj2', n_obj2s)
        ])
        
        labels = np.array([
            label_dict[shape] for shape in shapes
        ])

        shape_ids = np.concatenate([
            np.array(range(n_circles)),
            np.array(range(n_rectangles)),
            np.array(range(n_triangles)),
            np.array(range(n_obj1s)),
            np.array(range(n_obj2s))
        ])
        
        idx = np.array(range(n_rows))

        label_df = pd.DataFrame(data=labels, columns=['l_circ', 'l_rect', 'l_tri'])
        df = pd.DataFrame({'index':idx, 'shape':shapes, 'shape_id':shape_ids}, columns=['index', 'shape', 'shape_id'])
        df['l_circ'] = label_df['l_circ']
        df['l_rect'] = label_df['l_rect']
        df['l_tri'] = label_df['l_tri']

        df.to_csv(f'{self.root_dir}/shapes.csv', sep=',', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Synthetic Shape Themed Dataset")
    parser.add_argument('-o', '--root-directory', type=str, default='./data', help='The root directory for the generated dataset')
    parser.add_argument('-n', '--num-items', type=int, default=1000, help='The number of each individual class to be generated.')
    parser.add_argument('-s', '--img-shape', type=image_shape, default=(32,32), help='The width and height of the generated images')

    args = parser.parse_args()
    root_dir = args.root_directory
    num_items = args.num_items
    width, height = args.img_shape

    generator = DataGenerator(root_dir, (width, height), num_items)
    generator.generate_dataset()