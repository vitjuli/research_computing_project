import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NN Interpolator',
  description: '5D Neural Network Interpolation Tool',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  )
}
