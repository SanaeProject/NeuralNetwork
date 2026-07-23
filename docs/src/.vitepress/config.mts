import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MatrixDocs",
  description: "Matrix document",
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' }
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
            { text: 'Home', link: '/' },
            { text: 'コンストラクタ', link: '/constructors' },
            { text: 'ユーティリティ', link: '/utilities' },
            { text: 'イテレータ', link: '/iterators' },
            { text: '操作', link: '/operations' },
            { text: '変換', link: '/transforms' },
          ]
        }
      ],

      socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
