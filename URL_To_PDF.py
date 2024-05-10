import pdfkit
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Convert a URL to PDF')

# Add the arguments
parser.add_argument('Url', metavar='url', type=str, help='The URL to convert')
parser.add_argument('pdf_output', metavar='pdf_output', type=str, help='The PDF to output')

# Parse the arguments
args = parser.parse_args()

path = '../../../Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path)
url1 = args.Url
pdf = args.pdf_output
pdfkit.from_url(url1, pdf, configuration=config)