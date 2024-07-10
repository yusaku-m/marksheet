class PDFTools():
    def __init__(self):
        pass

    def get_list(self, terget_dir="Current", extension = "pdf"):
        """Return the file path list of specified extensions"""
        from glob import glob; import os
        if terget_dir == "Current":
            terget_dir = os.getcwd()
        
        list = []
        for file_name in sorted(glob(f'{terget_dir}/*.{extension}')):
            print(file_name)
            list.append(file_name)
        return list

    def make_from_img(self, img_list, pdf_path, direction="vertical", color="RGB"):
        """color:L is grayscale. 1 is monoscale."""
        import numpy as np
        from PIL import Image
        import cv2

        def pil2cv(image):
            ''' PIL型 -> OpenCV型 '''
            new_image = np.array(image, dtype=np.uint8)
            if new_image.ndim == 2:  # モノクロ
                pass
            elif new_image.shape[2] == 3:  # カラー
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            elif new_image.shape[2] == 4:  # 透過
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
            return new_image

        def cv2pil(image):
            ''' OpenCV型 -> PIL型 '''
            new_image = image.copy()
            if new_image.ndim == 2:  # モノクロ
                pass
            elif new_image.shape[2] == 3:  # カラー
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            elif new_image.shape[2] == 4:  # 透過
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
            new_image = Image.fromarray(new_image)
            return new_image

        pages = []

        for img_path in img_list:
            print(img_path)
            img = Image.open(img_path)
            img = pil2cv(img)
        
            h, w, _c = img.shape
            if (direction=="vertical" and h<w) or (direction=="horizontal" and w<h):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            multiple = 700 / img.shape[1]
            """
            while True:
                cv2.imshow("check", cv2.resize(img , (int(img.shape[1] * multiple), int(img.shape[0] * multiple))))
                cv2.waitKey(1)
                check = input("Is it OK? yes⇒1, no⇒0 :")
                if check == "1":
                    break
                else:
                    img = cv2.rotate(img, cv2.ROTATE_180)
            """
            cv2.imshow("check", cv2.resize(img , (int(img.shape[1] * multiple), int(img.shape[0] * multiple))))
            cv2.waitKey(1)
            img = cv2pil(img)
            pages.append(img.convert(color)) 
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

    def view_info(self, path):
        from PyPDF2 import PdfFileReader, PdfFileWriter
        input = open(path, 'rb')
        reader = PdfFileReader(input)
        print(type(reader.documentInfo))
        print(isinstance(reader.documentInfo, dict))
        for k in reader.documentInfo.keys():
            print(k, ':', reader.documentInfo[k])
        
    def concat(self, list, output_filename="merged_all_pages", output_dir="Current"):
        from PyPDF2 import PdfFileMerger; import os

        if output_dir == "Current":
            output_dir = os.getcwd()

        merger = PdfFileMerger()

        for file_path in list:
            print(file_path)
            merger.append(file_path)
        merger.write(f"{output_dir}/{output_filename}.pdf")

        merger.close()

    
    def resize(self, path):
        from PyPDF2 import PdfFileReader, PdfFileWriter
        PX_72DPI_A4 = (595, 842)
        input = open(path, 'rb')
        reader = PdfFileReader(input)
        writer = PdfFileWriter()

        for i in range(reader.getNumPages()):
            page = reader.getPage(i)
            #p_size = page.mediaBox; p_width = p_size.getWidth(); p_height = p_size.getHeight(); print(f"{p_width}, {p_height}")
            page.scale_to(PX_72DPI_A4[0], PX_72DPI_A4[1])
            writer.addPage(page)
        print('saving')
        output = open("output.pdf", 'wb')
        writer.write(output)
        print('saved')
        input.close(); output.close()

    def paf_compress(self, path):
        import os
        import sys
        import subprocess
        import configparser
        file = path
        fname = os.path.basename(file)
        dname = os.path.dirname(file)
        outfile_name = fname.replace('.pdf', '_min.pdf')

        subprocess.check_output([gs_path
            ,'-sDEVICE=pdfwrite'
            ,'-dCompatibilityLevel=1.4'
            ,r'-dPDFSETTINGS=/screen'
            ,'-dBATCH'
            ,'-dNOPAUSE'
            ,'-dQUIET'
            ,f'-sOUTPUTFILE={dname}\\{outfile_name}'
            , file
            ])

    def add_pagelabel(self, path, label = [], start_page=1, end_direction = "bottom", end_from=10):
        """
        各ページにlabelを付与する。
        end_from:指定終端からラベルを配置するまでの長さをmmで指定。塗りつぶしはその２倍行われる。
        """
        import io
        from PyPDF2 import PdfFileReader, PdfFileWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase import pdfmetrics

        def get_page_size(page):
            """
            既存PDFからページサイズ（幅, 高さ）を取得する
            """
            page_box = page.mediaBox
            width = page_box.getUpperRight_x() - page_box.getLowerLeft_x()
            height = page_box.getUpperRight_y() - page_box.getLowerLeft_y()
            return float(width), float(height)

        def create_page_number_pdf(canvas, page_size, label, end_direction = "bottom", end_from=10):
            """
            ラベルだけのPDFを作成
            """
            canvas.setPageSize(page_size)
            canvas.setFont('Times-Roman', 14)
            #一旦塗りつぶし
            canvas.setFillColorRGB(1,1,1) #choose fill colour
            if end_direction=="top":
                xs=0; xe=210; ys=297-end_from*2; ye=297; yc=297-end_from
            elif end_direction=="bottom":
                xs=0; xe=210; ys=0; ye=end_from*2; yc=end_from
            else:
                print("warning: end_direction can define 'top' or 'bottom' only.")
            canvas.rect(xs*mm,ys*mm,xe*mm,ye*mm, stroke=False, fill=1) #draw rectangle
            canvas.setFillColorRGB(0,0,0) #choose fill colour
            canvas.drawCentredString(page_size[0] / 2.0, yc, str(label))
            canvas.showPage()

        fi = open(path, 'rb')
        pdf_reader = PdfFileReader(fi)
        pages_num = pdf_reader.getNumPages()
        pdf_writer = PdfFileWriter()
        
        bs = io.BytesIO()
        c = canvas.Canvas(bs)
        #print(c.getAvailableFonts())

        if label == []:
            label = range(1,10000)

        for i in range(start_page-1, pages_num):
            # 既存PDF
            pdf_page = pdf_reader.getPage(i)
            # PDFページのサイズ
            page_size = get_page_size(pdf_page)
            # ラベル付きPDF作成
            create_page_number_pdf(c, page_size, label[i], end_direction, end_from)
        c.save()

        # ラベルだけのPDFをメモリから読み込み（seek操作はPyPDF2に実装されているので不要）
        pdf_num_reader = PdfFileReader(bs)

        # 既存PDFに１ページずつラベルを付ける
        for i in range(0, pages_num):
            # 既存PDF
            pdf_page = pdf_reader.getPage(i)
            # ラベルだけのPDF
            pdf_num = pdf_num_reader.getPage(i)

            # ２つのPDFを重ねる
            pdf_page.mergePage(pdf_num)
            pdf_writer.addPage(pdf_page)

        # ラベルを付けたPDFを保存
        fo = open("addpage.pdf", 'wb')
        pdf_writer.write(fo)

        bs.close(); fi.close(); fo.close()

    def split_odd_even(self, path):
        from PyPDF2 import PdfFileReader, PdfFileWriter
        input = open(path, 'rb')
        reader = PdfFileReader(input)
        
        odd_writer = PdfFileWriter()
        even_writer = PdfFileWriter()
        
        for i in range(reader.getNumPages()):
            page = reader.getPage(i)
            if i % 2 == 0:
                odd_writer.addPage(page)
            else:
                even_writer.addPage(page)
        print('saving')
        odd_output = open("odd.pdf", 'wb'); even_output = open("even.pdf", 'wb')
        odd_writer.write(odd_output); even_writer.write(even_output)
        print('saved')
        input.close(); odd_output.close(); even_output.close()

if __name__ == "__main__":
    """
    カレントディレクトリと同じ階層にあるフォルダ内のpdfを全て結合して生成する
    """

    from glob import glob
    import os

    pdf = PDFTools()

    dir_path = "./"
    dir_list = [
        f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
    ]

    for dir in dir_list:

        buf_list = sorted(glob(f"{dir}/**/*.pdf"))
        print(buf_list)

        pdf.concat(buf_list, 
                dir,
                dir_path)